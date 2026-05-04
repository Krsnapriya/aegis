Install in editable mode from the repo root:

```bash
pip install -e ./plutchik_erc_data
```

Load the bundled default CSV path (override with `PLUTCHIK_ERC_CSV`):

```python
from plutchik_erc_data import load_dataset
df = load_dataset(split="train")
```
