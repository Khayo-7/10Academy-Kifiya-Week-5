import datasets

_CITATION = """\
Your citation here.
"""

_DESCRIPTION = """\
Your dataset description here.
"""

class ConllDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                    # "ner_tags": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Extract file paths from self.config.data_files
        files = {
            "train": self.config.data_files["train"],
            "validation": self.config.data_files["validation"],
            "test": self.config.data_files["test"],
        }        

        # files = dl_manager.download_and_extract(files)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": files["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        # Ensure filepath is a string
        if isinstance(filepath, list):
            filepath = filepath[0]
        with open(filepath, encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split()
                    current_tokens.append(token)
                    current_tokens.append(0)
                    # current_labels.append(label_to_id.get(label, 0))
                    # current_labels.append(label)
                    # current_labels.append(int(label))
                else:
                    if current_tokens:
                        yield {
                            "tokens": current_tokens,
                            "ner_tags": current_labels,
                        }
                        current_tokens = []
                        current_labels = []
            if current_tokens:
                yield {
                    "tokens": current_tokens,
                    "ner_tags": current_labels,
                }