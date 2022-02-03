# coding=utf-8
# Copyright 2022 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""TODO"""

import datasets


_CITATION = """\
TODO
"""

_DESCRIPTION = """\
TODO
"""

_BASE_URL_TRAIN_DEV = "https://raw.githubusercontent.com/impresso/CLEF-HIPE-2020/master/data/training-v1.2/"


_URLs = {
    "EN": {
        "dev": _BASE_URL_TRAIN_DEV + "en/HIPE-data-v1.2-dev-en.tsv?raw=true"
    },  # English only has dev
    "DE": {
        "dev": _BASE_URL_TRAIN_DEV + "de/HIPE-data-v1.2-dev-de.tsv?raw=true",
        "train": _BASE_URL_TRAIN_DEV + "de/HIPE-data-v1.2-train-de.tsv?raw=true",
    },
    "FR": {
        "dev": _BASE_URL_TRAIN_DEV + "fr/HIPE-data-v1.2-dev-fr.tsv?raw=true",
        "train": _BASE_URL_TRAIN_DEV + "fr/HIPE-data-v1.2-train-fr.tsv?raw=true",
    },
}


class HIPE2020Config(datasets.BuilderConfig):
    """BuilderConfig for HIPE2020"""

    def __init__(self, data_urls, **kwargs):
        """BuilderConfig for HIPE2020.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HIPE2020Config, self).__init__(**kwargs)
        self.data_urls = data_urls


class HIPE2020(datasets.GeneratorBasedBuilder):
    """HIPE2020 dataset."""

    BUILDER_CONFIGS = [
        HIPE2020Config(
            name="en",
            data_urls=_URLs["EN"],
            version=datasets.Version("1.0.0"),
            description="HIPE dataset covering English",
        ),
        HIPE2020Config(
            name="de",
            data_urls=_URLs["DE"],
            version=datasets.Version("1.0.0"),
            description="HIPE dataset covering German",
        ),
        HIPE2020Config(
            name="fr",
            data_urls=_URLs["FR"],
            version=datasets.Version("1.0.0"),
            description="HIPE dataset covering French",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "NE_COARSE_LIT": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-loc",
                                "B-org",
                                "B-pers",
                                "B-prod",
                                "B-time",
                                "I-loc",
                                "I-org",
                                "I-pers",
                                "I-prod",
                                "I-time",
                            ]
                        )
                    ),
                    "NE_COARSE_METO_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-loc",
                                "B-org",
                                "B-pers",
                                "B-prod",
                                "B-time",
                                "I-loc",
                                "I-org",
                                "I-pers",
                                "I-prod",
                                "I-time",
                            ]
                        )
                    ),
                    "no_space_after": datasets.Sequence(datasets.Value("bool")),
                    "end_of_line": datasets.Sequence(datasets.Value("bool")),
                }
            ),
            supervised_keys=None,
            homepage="TODO",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(self.config.data_urls)
        if self.config.name != "en":
            data_files = {
                "train": downloaded_files["train"],
                "dev": downloaded_files["dev"],
            }
        else:
            data_files = {"dev": downloaded_files["dev"]}
        if self.config.name == "en":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": data_files["dev"]},
                ),
                # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}), # TODO add test splits
            ]

        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_files["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": data_files["dev"]},
                ),
            ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            NE_COARSE_LIT_tags = []
            NE_COARSE_METO_tags = []
            no_space_after = []
            end_of_line = []
            for line in f:
                if line.startswith(
                    "TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC"
                ):
                    continue
                if line.startswith("#") or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "NE_COARSE_LIT": NE_COARSE_LIT_tags,
                            "NE_COARSE_METO_tags": NE_COARSE_METO_tags,
                            "no_space_after": no_space_after,
                            "end_of_line": end_of_line,
                        }
                        guid += 1
                        tokens = []
                        NE_COARSE_LIT_tags = []
                        NE_COARSE_METO_tags = []
                        no_space_after = []
                        end_of_line = []
                else:
                    # HIPE 2020 tokens are tab separated
                    splits = line.split(
                        "\t"
                    )  # TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
                    tokens.append(splits[0])
                    NE_COARSE_LIT_tags.append(splits[1])
                    NE_COARSE_METO_tags.append(splits[2])
                    misc = splits[-1]
                    is_space = "NoSpaceAfter" in misc
                    is_end_of_line = "EndOfLine" in misc
                    no_space_after.append(is_space)
                    end_of_line.append(is_end_of_line)

            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "NE_COARSE_LIT": NE_COARSE_LIT_tags,
                "NE_COARSE_METO_tags": NE_COARSE_METO_tags,
                "no_space_after": no_space_after,
                "end_of_line": end_of_line,
            }
