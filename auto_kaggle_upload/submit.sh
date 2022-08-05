mkdir ~/.kaggle 
cp data/kaggle.json ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json
python auto_kaggle_upload/submit.py