if hasattr(__builtins__,'__IPYTHON__'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

for i in tqdm(range(10000000)):
    a=5
    