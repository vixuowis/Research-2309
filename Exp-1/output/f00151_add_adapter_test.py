from f00151_add_adapter import *
import unittest
from transformers import AutoModel
from transformers.adapters import AdapterConfig


class TestAddAdapter(unittest.TestCase):
    def test_add_adapter(self):
        model = AutoModel.from_pretrained('bert-base-uncased')
        adapter_name = 'my_adapter'
        
        add_adapter(model, adapter_name)
        
        self.assertIn(adapter_name, model.config.adapters.adapter_list)


if __name__ == '__main__':
    unittest.main()
