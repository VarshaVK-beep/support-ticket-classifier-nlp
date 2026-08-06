from datasets import load_dataset

dataset = load_dataset('Tobi-Bueck/customer-support-tickets')
print(dataset)
print(dataset['train'][0])
