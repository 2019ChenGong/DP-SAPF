#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable Diffusion Selective Fine-tuning JSON to LaTeX Table Converter
Author: Kecen Li
Date: 2026-02-05

Usage:
    python json_to_latex_table.py \
        --input configs/10p.json configs/30p.json configs/70p.json configs/100p.json \
        --output table.tex \
        --caption "Selective fine-tuning configuration across selection ratios." \
        --label "tab:selective_finetuning"
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class SDAttentionTableGenerator:
    """Convert SD attention selection JSONs to LaTeX table"""
    
    # 完整的module列表（按Down/Mid/Up分组）
    MODULE_ORDER = [
        # Down Blocks (6 modules)
        "down_blocks.0.attentions.0",
        "down_blocks.0.attentions.1",
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.0",
        "down_blocks.2.attentions.1",
        # Mid Block (1 module)
        "mid_block.attentions.0",
        # Up Blocks (9 modules)
        "up_blocks.1.attentions.0",
        "up_blocks.1.attentions.1",
        "up_blocks.1.attentions.2",
        "up_blocks.2.attentions.0",
        "up_blocks.2.attentions.1",
        "up_blocks.2.attentions.2",
        "up_blocks.3.attentions.0",
        "up_blocks.3.attentions.1",
        "up_blocks.3.attentions.2",
    ]
    
    # 简化名称映射
    SHORT_NAMES = {
        "down_blocks.0.attentions.0": "D0-A0",
        "down_blocks.0.attentions.1": "D0-A1",
        "down_blocks.1.attentions.0": "D1-A0",
        "down_blocks.1.attentions.1": "D1-A1",
        "down_blocks.2.attentions.0": "D2-A0",
        "down_blocks.2.attentions.1": "D2-A1",
        "mid_block.attentions.0": "M-A0",
        "up_blocks.1.attentions.0": "U1-A0",
        "up_blocks.1.attentions.1": "U1-A1",
        "up_blocks.1.attentions.2": "U1-A2",
        "up_blocks.2.attentions.0": "U2-A0",
        "up_blocks.2.attentions.1": "U2-A1",
        "up_blocks.2.attentions.2": "U2-A2",
        "up_blocks.3.attentions.0": "U3-A0",
        "up_blocks.3.attentions.1": "U3-A1",
        "up_blocks.3.attentions.2": "U3-A2",
    }
    
    def __init__(self, json_files: List[str]):
        """
        Args:
            json_files: List of JSON file paths, format: "configs/10p.json" or "70p.json"
        """
        self.json_files = json_files
        self.configs = {}  # {ratio: {module_path: {attn1/attn2: [projections]}}}
        self.ratios = []
        
    def extract_ratio_from_filename(self, filepath: str) -> str:
        """从文件名提取比例，如 '10p.json' -> '10%'"""
        filename = Path(filepath).name
        # 匹配数字 + p 或 % 的模式
        match = re.search(r'(\d+)(?:p|%)', filename)
        if match:
            return f"{match.group(1)}%"
        else:
            # 默认使用文件名（不含扩展名）
            return Path(filename).stem.replace('p', '%')
    
    def parse_json(self, filepath: str) -> Dict[str, Dict[str, List[str]]]:
        """
        解析JSON，返回结构: {module_base_path: {attn1/attn2: [q,k,v]}}
        
        Example input:
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": ["k", "q", "v"]
        
        Output:
            {
                "down_blocks.0.attentions.0": {
                    "attn1": ["k", "q", "v"],
                    "attn2": ["k", "v"]
                }
            }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = defaultdict(dict)
        
        for full_path, projections in data.items():
            # 提取基础路径（去掉.transformer_blocks.0.attnX.processor）
            base_path = re.sub(r'\.transformer_blocks\.0\.attn[12]\.processor$', '', full_path)
            
            # 提取是attn1还是attn2
            attn_type_match = re.search(r'\.attn([12])\.processor$', full_path)
            if attn_type_match:
                attn_type = f"attn{attn_type_match.group(1)}"
                result[base_path][attn_type] = sorted(projections)  # 排序保证一致性
        
        return dict(result)
    
    def format_cell(self, attn1_projs: Optional[List[str]], attn2_projs: Optional[List[str]]) -> str:
        """
        格式化单元格内容
        
        Args:
            attn1_projs: attn1的投影矩阵列表，如 ["v"] 或 None
            attn2_projs: attn2的投影矩阵列表，如 ["k", "v"] 或 None
        
        Returns:
            格式化字符串，如 "v/kv" 或 "--/--"
        """
        def proj_to_str(projs):
            if not projs:
                return "--"
            # 按q,k,v顺序排序并拼接
            return ''.join(sorted(projs))
        
        attn1_str = proj_to_str(attn1_projs)
        attn2_str = proj_to_str(attn2_projs)
        
        return f"{attn1_str}/{attn2_str}"
    
    def load_configs(self):
        """加载所有JSON配置并按比例排序"""
        config_dict = {}
        for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            filepath = "exp/lora_cifar10_32_4096bs_1ksteps_eps10/lora_k4q4v4o4_base_top{}_fs5_finegrained_0.0005/tuning_layers.json".format(ratio)
            # ratio = self.extract_ratio_from_filename(filepath)
            config = self.parse_json(filepath)
            config_dict[int(ratio*100)] = config
        
        # 按比例数值排序（10% < 30% < 70% < 100%）
        def ratio_key(r):
            return int(r.replace('%', ''))
        
        self.ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.ratios = [int(x*100) for x in self.ratios]
        self.configs = {r: config_dict[r] for r in self.ratios}
    
    def generate_table_row(self, ratio: str) -> str:
        """生成表格的一行（对应一个比例）"""
        config = self.configs[ratio]
        cells = []
        
        for module_base in self.MODULE_ORDER:
            module_data = config.get(module_base, {})
            attn1_projs = module_data.get('attn1')
            attn2_projs = module_data.get('attn2')
            
            cell_content = self.format_cell(attn1_projs, attn2_projs)
            cells.append("\\texttt{"+cell_content+"}")
        
        # 加粗当前行的比例标签
        ratio_label = f"{ratio}\\%"
        row = f"{ratio_label} & " + " & ".join(cells) + " \\\\"
        return row
    
    def generate_latex_table(
        self,
        caption: str = "Selective fine-tuning configuration across selection ratios.",
        label: str = "tab:selective_finetuning",
        note: Optional[str] = None
    ) -> str:
        """
        生成完整的LaTeX表格代码
        
        Args:
            caption: 表格标题
            label: LaTeX标签
            note: 表格下方的注释说明
        """
        # 表头：简化名称
        header_cells = [self.SHORT_NAMES[m] for m in self.MODULE_ORDER]
        
        # 列格式：l | ccccccc | c | ccccccccc
        # Down: 6列, Mid: 1列, Up: 9列
        col_format = "l|" + "c" * 6 + "|c|" + "c" * 9
        
        # 生成LaTeX表格
        latex = []
        latex.append("\\begin{table*}[t]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append("\\resizebox{\\textwidth}{!}{")
        latex.append("\\begin{tabular}{" + col_format + "}")
        latex.append("\\toprule")
        
        # 多级表头
        latex.append("\\multirow{2}{*}{\\textbf{Ratio}} & ")
        latex.append("\\multicolumn{6}{c|}{\\textbf{Down Blocks}} & ")
        latex.append("\\multicolumn{1}{c|}{\\textbf{Mid Block}} & ")
        latex.append("\\multicolumn{9}{c}{\\textbf{Up Blocks}} \\\\")
        
        latex.append("\\cmidrule(lr){2-7} \\cmidrule(lr){8-8} \\cmidrule(l){9-17}")
        latex.append("& " + " & ".join(header_cells) + " \\\\")
        latex.append("\\midrule")
        
        # 数据行
        for ratio in self.ratios:
            latex.append(self.generate_table_row(ratio))
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("}")
        
        # 添加注释
        if note is None:
            note = (
                "Each cell shows fine-tuned projections for self-attention (attn1) "
                "and cross-attention (attn2) as \\texttt{attn1/attn2} "
                "(e.g., \\texttt{v/qkv} = fine-tune \\textbf{v} in attn1 and "
                "\\textbf{q,k,v} in attn2). ``--'' indicates frozen (not selected)."
            )
        latex.append(f"\\vspace{{2pt}}\\\\ \\small \\textit{{{note}}}")
        
        latex.append("\\end{table*}")
        
        return "\n".join(latex)
    
    def generate_mapping_table(self) -> str:
        """生成模块名称映射表（供附录使用）"""
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Module name mapping: simplified notation to full path}")
        latex.append("\\label{tab:module_mapping}")
        latex.append("\\begin{tabular}{ll}")
        latex.append("\\toprule")
        latex.append("\\textbf{Simplified} & \\textbf{Full Path} \\\\")
        latex.append("\\midrule")
        
        for full_path, short_name in self.SHORT_NAMES.items():
            latex.append(f"{short_name} & \\texttt{{{full_path}}} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def print_statistics(self):
        """打印每个配置的统计信息"""
        print("=" * 60)
        print("Configuration Statistics")
        print("=" * 60)
        
        for ratio in self.ratios:
            config = self.configs[ratio]
            total_processors = 0
            total_projections = 0
            
            for module_data in config.values():
                for projs in module_data.values():
                    total_processors += 1
                    total_projections += len(projs)
            
            print(f"{ratio} | Processors: {total_processors:3d} | Projections: {total_projections:3d}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SD attention selection JSONs to LaTeX table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python json_to_latex_table.py configs/*.json -o table.tex
  python json_to_latex_table.py 10p.json 30p.json 70p.json 100p.json --caption "My Table"
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input JSON files (e.g., 10p.json 30p.json 70p.json)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='attention_table.tex',
        help='Output LaTeX file (default: attention_table.tex)'
    )
    
    parser.add_argument(
        '-c', '--caption',
        default='Selective fine-tuning configuration across selection ratios.',
        help='Table caption'
    )
    
    parser.add_argument(
        '-l', '--label',
        default='tab:selective_finetuning',
        help='LaTeX label (default: tab:selective_finetuning)'
    )
    
    parser.add_argument(
        '-n', '--note',
        help='Custom note below the table'
    )
    
    parser.add_argument(
        '--mapping',
        action='store_true',
        help='Also generate module name mapping table'
    )
    
    parser.add_argument(
        '--stdout',
        action='store_true',
        help='Print to stdout instead of file'
    )
    
    args = parser.parse_args()
    
    # 生成表格
    generator = SDAttentionTableGenerator(args.input_files)
    generator.load_configs()
    
    # 打印统计信息
    generator.print_statistics()
    
    # 生成LaTeX
    latex_table = generator.generate_latex_table(
        caption=args.caption,
        label=args.label,
        note=args.note
    )
    
    # 生成映射表（可选）
    if args.mapping:
        mapping_table = generator.generate_mapping_table()
        latex_table = latex_table + "\n\n" + mapping_table
    
    # 输出
    if args.stdout:
        print(latex_table)
    else:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f"\n✓ LaTeX table saved to: {args.output}")
        
        if args.mapping:
            print(f"✓ Module mapping table also included")


if __name__ == '__main__':
    main()

# python vis_selection.py xx -o table.tex