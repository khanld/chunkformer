import re
from os import PathLike
from typing import Dict, List, Optional, Union

from chunkformer.text.base_tokenizer import BaseTokenizer, T
from chunkformer.utils.file_utils import read_non_lang_symbols, read_symbol_table


class CharTokenizer(BaseTokenizer):

    def __init__(
        self,
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = "",
        unk="<unk>",
    ) -> None:
        self.non_lang_syms_pattern = None
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        if not isinstance(symbol_table, Dict):
            self._symbol_table = read_symbol_table(symbol_table)
        else:
            # symbol_table = {"我": 1, "是": 2, "{NOISE}": 3}
            self._symbol_table = symbol_table
        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        self.char_dict = {v: k for k, v in self._symbol_table.items()}
        self.split_with_space = split_with_space
        self.connect_symbol = connect_symbol
        self.unk = unk

    def text2tokens(self, line: str) -> List[T]:
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w.strip() for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.split_with_space:
                    part = part.split(" ")
                for ch in part:
                    if ch == " ":
                        ch = "▁"
                    tokens.append(ch)
        return tokens

    def tokens2text(self, tokens: List[T]) -> str:
        # Convert tokens to strings if they're bytes
        str_tokens = []
        for token in tokens:
            if isinstance(token, bytes):
                str_tokens.append(token.decode("utf-8"))
            else:
                str_tokens.append(token)
        return self.connect_symbol.join(str_tokens)

    def tokens2ids(self, tokens: List[T]) -> List[int]:
        ids = []
        for ch in tokens:
            if ch in self._symbol_table:
                ids.append(self._symbol_table[ch])
            elif self.unk in self._symbol_table:
                ids.append(self._symbol_table[self.unk])
        return ids

    def ids2tokens(self, ids: List[int]) -> List[T]:
        content = [self.char_dict[w] for w in ids]
        return content

    def vocab_size(self) -> int:
        return len(self.char_dict)

    @property
    def symbol_table(self) -> Dict[T, int]:
        return self._symbol_table  # type: ignore
