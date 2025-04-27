import kagglehub

# Download latest version
path = kagglehub.dataset_download("visualcomments/russian-civil-code-all-parts")

print("Path to dataset files:", path)