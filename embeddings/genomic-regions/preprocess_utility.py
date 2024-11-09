
import gzip
import pandas as pd
import csv

def read_fasta(fasta_file):
    sequences = {}
    sequence_name = ''
    sequence_data = []
    
    file_handle=open(fasta_file, 'r')
    file_handle.seek(0)

    for line in file_handle:
            line = line.strip()
            if line.startswith('>'):
                if sequence_name:
                    sequences[sequence_name] = ''.join(sequence_data)
                sequence_name = line[1:]  # Remove the '>' character
                sequence_data = []
            else:
                sequence_data.append(line)
        
    if sequence_name:
            sequences[sequence_name] = ''.join(sequence_data)
    
    file_handle.close()
    
    return sequences





def append_rows_to_csv(csv_Filename, rows):
    with open(csv_Filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(rows)
        for row in rows:
            writer.writerow(row)
            



def load_embedding_file(csv_filename):

    df=pd.read_csv(csv_filename)
        
    column_names = [f'{i}' for i in range(1, df.shape[1])]
    column_names.extend([ 'y'])
    
    df.columns = column_names
    return df


def swapfirst2last(df):
    df = df[df.columns[1:].tolist() + df.columns[:1].tolist()]
    return df

def  preprocess_datafile(data_filename):
    comp = {'A':1, 'C':2, 'G':3, 'T':4}
    
    max_length= 128 # 186
    
    # Open the gzipped file and read it into a DataFrame
    
    with gzip.open(data_filename, 'rt') as f:
        df = pd.read_csv(f, sep='\t')  # Automatically detects header from the file
    
    
    df['CHROM'] = df['CHROM'].str.replace('chr', '', regex=False)
    
    # df['REF']='A'
    # df['ALT']='A'
    df['SIZE']=max_length
    
    df = df.rename(columns={'Percentage':'y', 'FROM':'START'})
    df = df.drop(['TO','Coverage'], axis=1)
    
    
    cols = df.columns.tolist()
    
    # Move the 3rd column to the last position
    cols.append(cols.pop(2))
    
    # Reorder the DataFrame
    df = df[cols]
    df=df[~df['CHROM'].str.contains('KI',na=False)]
    df=df[~df['CHROM'].str.contains('GL',na=False)]
    df=df[~df['CHROM'].str.contains('M',na=False)]
    return df



def  preprocess_home_sapiens_datafile(data_filename):
    
    df = pd.read_csv(data_filename, sep='\t')  # ,header=None
    df['ROWID'] = df.index
    df.columns = df.columns.str.upper()

    # df['cluster'].value_counts()
    cluster_dict = {
        'first_exon': 0,
        'first_intron': 1,
        'first_three_prime_UTR': 2,
        'first_five_prime_UTR': 3,
        'ncRNA_gene': 4,
        'pseudogene': 5,
        'smallRNA': 6
    }
    df['y'] = df['CLUSTER'].map(cluster_dict)

    # filtered_df = df[df['chromosome'].str.contains('KI')]
    df = df[~df['CHROM'].str.contains('KI')]
    df = df[~df['CHROM'].str.contains('GL')]
    df = df[~df['CHROM'].str.contains('M')]

    df.drop_duplicates(subset=['CHROM', 'START'], keep='first', inplace=True)
    
    return df