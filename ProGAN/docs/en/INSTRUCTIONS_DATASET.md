# 📥 Instructions to Download the Dataset

## Option 1: Download from Kaggle (Recommended)

### Step 1: Create a Kaggle account
1. Go to [Kaggle.com](https://www.kaggle.com/)
2. Create a free account if you don't have one

### Step 2: Download the dataset
1. Go to the dataset page: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
2. Click the **"Download"** button (⚠️ The file is ~5GB)
3. The downloaded file is named `archive.zip`

### Step 3: Place the file
1. Move the `archive.zip` file to the folder:
   ```
   OCT_GANs\ProGAN\
   ```
2. **DO NOT unzip it manually** — the script will extract it automatically

### Step 4: Run the preparation script
```powershell
cd "c:\Users\joaco\Documents\Programming\OCT\preexisting_repositorys\OCT_GANs\ProGAN"
python download_dataset.py
```

The script will:
- ✅ Automatically detect the zip file
- ✅ Extract ONLY the DRUSEN images (~8k images)
- ✅ Organize them into the correct folder structure
- ✅ Clean up temporary files

---

## Option 2: Use the Kaggle API (Faster)

### Requirements:
1. A Kaggle account
2. A Kaggle API token

### Steps:

1. **Get Kaggle credentials**:
   - Go to https://www.kaggle.com/settings
   - Scroll to the "API" section
   - Click "Create New API Token"
   - A `kaggle.json` file will be downloaded

2. **Configure the Kaggle API**:
   ```powershell
   # Install kaggle
   pip install kaggle

   # Create credentials directory (if it doesn't exist)
   mkdir $env:USERPROFILE\.kaggle -Force

   # Copy kaggle.json to the credentials directory
   Copy-Item "C:\path\to\downloaded\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
   ```

3. **Download the dataset automatically**:
   ```powershell
   cd "c:\Users\joaco\Documents\Programming\OCT\preexisting_repositorys\OCT_GANs\ProGAN"

   # Download the dataset
   kaggle datasets download -d paultimothymooney/kermany2018

   # Run the preparation script
   python download_dataset.py
   ```

---

## 🔍 Verify everything is ready

After running the script, you should see:
```
✅ Done! You have XXXX DRUSEN images in C:\...\ProGAN\data\OCT2017\train\DRUSEN
```

The directory structure will be:
```
ProGAN/
├── data/
│   └── OCT2017/
│       └── train/
│           └── DRUSEN/
│               ├── image1.jpeg
│               ├── image2.jpeg
│               └── ...
├── download_dataset.py
└── progan_local.py
```

---

## ⚠️ Troubleshooting

### "No zip file found"
- Check that `archive.zip` is in the `ProGAN/` folder
- The file might be named `archive.zip`, `kermany2018.zip`, or `OCT2017.zip`

### "Error extracting"
- Make sure the zip file is not corrupted
- Re-download if necessary
- Ensure you have enough free disk space (~10GB)

### "No DRUSEN images found"
- The script searches automatically for folders with "DRUSEN" in the name
- Verify you downloaded the correct dataset

---

## 🚀 Next step

Once the dataset is prepared, run:
```powershell
python progan_local.py
```

This will start training or generate images according to your configuration.
