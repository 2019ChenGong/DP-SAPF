eval "$(conda shell.bash hook)"

conda activate dpimagebench;
mkdir dataset;
cd data;
python preprocess_dataset.py --data_name cifar10; cd ..
conda deactivate;