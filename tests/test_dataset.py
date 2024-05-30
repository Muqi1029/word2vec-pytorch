import pytest

from dataset.treebank import StanfordSentiment

class TestDataset:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.data = StanfordSentiment("stanfordSentimentTreebank")
    
    def test_get_properties(self):
        print(len(self.data.tokens()))
    
    def test_get_window(self):
        print(len(self.data.tokens()))
        nums = 5
        for i in range(nums):
            print(self.data.getRandomContext())
            
    
    def test_get_negative_sample(self):
        pass