import pytest
from run import main, parse_args

class TestRun:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        test_args = ['--data_dir', 'stanfordSentimentTreebank',
                     '--lr', '0.1',
                     '--num_epochs', '1']
        self.args = parse_args(test_args)
        

    def test_run(self):
        print("args:", self.args)
        main(self.args)