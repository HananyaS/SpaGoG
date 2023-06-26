from datasets.graphDataPair import GraphDataPair
from datasets.tabDataset import TabDataset

from copy import deepcopy


class GraphDataset(TabDataset):
    def __init__(self, train_mask: list = None, val_mask: list = None, test_mask: list = None, **kwargs):
        super(GraphDataset, self).__init__(**kwargs)

        train_mask = train_mask if train_mask is not None else ["train"] * self.train.get_num_samples()
        val_mask = val_mask if val_mask is not None else ["val"] * self.val.get_num_samples()
        test_mask = test_mask if test_mask is not None else ["test"] * self.test.get_num_samples()

        self.mask = train_mask + val_mask + test_mask

        self.train_idx = list(
            filter(lambda i: self.mask[i] == "train", list(range(len(self.mask))))
        )

        self.data = deepcopy(self.train)
        self.data += self.val
        self.data += self.test

    @classmethod
    def from_tab(cls, tab_data: TabDataset, **kwargs):
        train = GraphDataPair.from_tab(tab_data=tab_data.train, **kwargs)

        val = (
            None
            if not tab_data.val_exists
            else GraphDataPair.from_tab(tab_data=tab_data.val, **kwargs)
        )

        test = (
            None
            if not tab_data.test_exists
            else GraphDataPair.from_tab(tab_data=tab_data.test, **kwargs)
        )

        graph_dataset = cls(
            train=train,
            val=val,
            test=test,
            normalize=False,
            name=f"{tab_data.name} - graph",
        )

        graph_dataset.normalized = tab_data.normalized

        return graph_dataset

    def get_graph_data(self):
        self.data.edges_to_lst(inplace=True)
        return self.data

    def get_train_loader(self, **kwargs):
        return self.train.to_loader(**kwargs)

    def get_val_loader(self, **kwargs):
        return self.val.to_loader(**kwargs)

    def get_test_loader(self, **kwargs):
        return self.test.to_loader(**kwargs)
