import os
import argparse

def download_files(download_path):
    # Create directories if they don't exist
    os.makedirs(os.path.join(download_path, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(download_path, 'pmids'), exist_ok=True)
    os.makedirs(os.path.join(download_path, 'pubmed_content'), exist_ok=True)

    for i in range(37):
        print(f"Downloading files for chunk {i}...")
        
        # Download embeddings
        os.system(f"wget -P {os.path.join(download_path, 'embeddings')} https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/embeds_chunk_{i}.npy")
        
        # Download corresponding PMIDs
        os.system(f"wget -P {os.path.join(download_path, 'pmids')} https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/pmids_chunk_{i}.json")
        
        # Download PMID content
        os.system(f"wget -P {os.path.join(download_path, 'pubmed_content')} https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/pubmed_chunk_{i}.json")

        print(f"Completed downloading files for chunk {i}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files to a specified directory.')
    parser.add_argument('--path', type=str, default='PMData', help='The directory to download files to (default: PMData)')
    args = parser.parse_args()
    
    download_files(args.path)


#python download_pubmed_index.py --path /pasteur/data/PubMed