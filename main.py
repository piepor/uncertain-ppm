import argparse
import yaml
from shutil import copyfile
from pathlib import Path
from inference import InferenceTFEnsemble, InferenceTFMonteCarlo
import tensorflow.keras as tfk
import training
from dataprocessing import DataProcessorTF, Stage, Mode
from utilities import helpdesk, bpic2012, bpic2013, road_traffic_fines
from utilities.directories import create_models_dir
from visualizing import ResultsPlotter

split_map = {
        Stage.TRAIN: "train[:70%]",
        Stage.VALI: "train[70%:85%]",
        Stage.TEST: "train[85%:]"}

dataset_utils_map = {
        "helpdesk": helpdesk.helpdesk_utils,
        "bpic2012": bpic2012.bpic2012_utils, 
        "bpic2013": bpic2013.bpic2013_utils,
        "road_traffic_fines": road_traffic_fines.road_traffic_fines_utils,
        }

stage_map = {Stage.TRAIN: 'Training set',
             Stage.VALI: 'Validation set',
             Stage.TEST: 'Test set'}

def main(dataset_name: str, mode: Mode, model_name: str="", uncertainty_type: str="", predict_case: str="", overconfidence_analysis: bool=False):
    if mode == Mode.TRAIN:
        with open('training.params', 'rb') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        features_path = Path('.') / 'features' / dataset_name / params['features-file']
        train_dataprocessor = DataProcessorTF(dataset_name, Stage.TRAIN, dataset_utils_map)
        train_dataprocessor.get_dataset_utils()
        train_dataprocessor.compute_features(features_path)
        train_dataprocessor.import_dataset(split_map)
        vali_dataprocessor = DataProcessorTF(dataset_name, Stage.VALI, dataset_utils_map)
        vali_dataprocessor.get_dataset_utils()
        vali_dataprocessor.compute_features(features_path)
        vali_dataprocessor.import_dataset(split_map)
        models_dir = create_models_dir(dataset_name)
        experiment_runner = training.ExperimentRunnerTFEnsemble(
                models_dir, train_dataprocessor, vali_dataprocessor, params)
        experiment_runner.run(params)
        copyfile(features_path, models_dir / 'features.params')

    elif mode == Mode.VISUALIZE:
        model_dir = Path('.') / 'models' / dataset_name / model_name
        figures_dir = Path('.') / 'saved_figures' / dataset_name 
        features_path = model_dir / 'features.params'
        test_dataprocessor = DataProcessorTF(dataset_name, Stage.TEST, dataset_utils_map)
        test_dataprocessor.get_dataset_utils()
        test_dataprocessor.compute_features(features_path)
        test_dataprocessor.import_dataset(split_map)
        models_id = [child for child in model_dir.iterdir() if child.is_dir()]

        if not models_id:
            raise FileNotFoundError(f"Models not present in directory: {str(model_dir)}")

        if uncertainty_type == 'MC': 
            model_path = models_id[0]
            model = tfk.models.load_model(f"./{str(model_path)}")
            inference_model = InferenceTFMonteCarlo(test_dataprocessor, model,
                                                    overconfidence_analysis=overconfidence_analysis,
                                                    model_dir=model_dir,
                                                    set_type=stage_map[Stage.TEST])
        elif uncertainty_type == 'ensemble':
            models = []
            for child in models_id:
                model = tfk.models.load_model(f"./{str(child)}")
                models.append(model)
            inference_model = InferenceTFEnsemble(test_dataprocessor, models,
                                                  overconfidence_analysis=overconfidence_analysis,
                                                  model_dir=model_dir,
                                                  set_type=stage_map[Stage.TEST])
        else:
            raise NameError(f"Uncertainty type {uncertainty_type} not available")

        inference_model.run(predict_case)
        results = inference_model.get_results()
        title_text= f"Model '{model_name}' - {uncertainty_type} - {stage_map[Stage.TEST]} - {dataset_name}"
        plotter = ResultsPlotter(results, title_text, figures_dir)
        plotter.reliability_diagram()
        plotter.accuracy_uncertainty()
        plotter.event_probability(test_dataprocessor.vocabularies['activity'])

        if predict_case:
            plotter.case_prediction(test_dataprocessor.vocabularies['activity'], uncertainty_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset considered to process', 
                        choices=['helpdesk', 'bpic2012', 'bpic2013', 'road_traffic_fines'])
    parser.add_argument('mode', help='Mode to launch the program: train the model or visualize results',
                        choices=['train', 'vis'])
    parser.add_argument('--model_dir', help='name of the directory where the models are stored',
                        default="", type=str)
    parser.add_argument('--uncertainty_type', help='how to inject stochasticity',
                        choices=['ensemble', 'MC'], default='MC', type=str)
    parser.add_argument('--predict_case', help='id of the case to predict',
                        default="", type=str)
    parser.add_argument('--overconfidence_analysis', help='Launch the analysis on overconfident wrong predictions',
                        default=False, type=bool)
    args = parser.parse_args()
    dataset = args.dataset
    modes = {'train': Mode.TRAIN, 'vis': Mode.VISUALIZE}
    main(dataset, modes[args.mode], args.model_dir, args.uncertainty_type, args.predict_case, args.overconfidence_analysis)
