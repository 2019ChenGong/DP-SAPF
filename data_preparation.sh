conda activate dpimagebench;
mkdir dataset;
cd data;
python preprocess_dataset.py --data_name cifar10_32; cd ..
conda deactivate;