from torch.utils.data import Dataset

class MultiAttributeDataset(Dataset):
    def __init__(self, attribute_names, attribute_class_counts):
        self.attribute_names = attribute_names
        self.attribute_count = len(attribute_names)
        self.attribute_class_counts = attribute_class_counts
        self.attribute_class_score_ranges = [
            range(
                range_start := sum(attribute_class_counts[0:attribute_index]),
                range_start + attribute_class_counts[attribute_index],
            )
            for attribute_index in range(self.attribute_count)
        ]
