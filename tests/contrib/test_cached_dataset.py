if __name__ == "__main__":
    n = 10

    class TestDataset(Dataset):
        def __getitem__(self, item):
            if item < 10:
                return (np.random.rand(5, 2), np.random.rand(1))
            raise IndexError("xxx")

        def __len__(self):
            return 10

    dataset = TestDataset()

    for i, data in enumerate(dataset):
        print(i, data)

    # with CachedDataset(dataset) as ms:
    ms = CachedDataset(dataset)
    print(len(ms))
    for _ in range(100):
        for i, data in enumerate(ms):
            print(i, data)
