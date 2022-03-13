from functions.generate_word_cloud import generate_word_cloud

csv_files = ['Fake', 'True']

for file in csv_files:
    generate_word_cloud(file)
