import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/song-popularity-dataset")

print("Path to dataset files:", path)