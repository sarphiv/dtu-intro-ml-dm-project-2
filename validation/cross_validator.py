from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from typing import List, Tuple, Union, Callable
import numpy as np
from joblib import Parallel, delayed

from validation.validation_model import ValidationModel
from validation.validation_result import ValidationResult

#List of indices
IdxList = np.ndarray
#Tuple of training and test indices
IdxTrainTest = Tuple[IdxList, IdxList]

#Loss function signature
LossFunc = Callable[[np.ndarray, np.ndarray], float]

#Identifier (outer loop ID, inner loop ID, model ID)
LId = Union[Tuple[int, int, int], int]


class CrossValidator:
    def __init__(self, 
                 n_outer: int, n_inner: int,
                 n_workers: int, verbose: bool = False,
                 randomize_seed: int = None) -> None:
        """
        n_outer: Set to 0 to do one-layer cross-validation
        n_workers: Amount of workers. Too high a value adds coordination overhead.
        """
        
        #Set up properties
        self.n_outer = n_outer
        self.n_inner = n_inner
        
        self.n_workers = n_workers
        self.verbose = verbose
        self.randomize_seed = randomize_seed
        


    @staticmethod
    def __create_idx_splits(n_data, n_outer, n_inner) -> List[Tuple[IdxTrainTest, List[IdxTrainTest]]]:
        """
        Returns: List of tuples of, 
            outer train-test split and 
            its associated inner train-test splits
        """
        #Store all indexes of data in order
        indexes = np.arange(n_data)


        #If one-layer, mark all data as training
        if n_outer == 0:
            outer_splits = [(indexes, np.empty(0))]
        #Else two-layer, create outer splits, 
        else:
            outer_splits = list(KFold(n_outer).split(indexes))


        #Splits to return
        splits = []
        splitter = KFold(n_inner)

        for outer_split in outer_splits:
            #Retrieve training part of outer split
            idx_train, _ = outer_split

            #Get inner train-test indexes
            inner_splits = [(idx_train[train], idx_train[test])
                            for train, test in splitter.split(idx_train)]

            #Save outer-inner splits
            splits.append((outer_split, inner_splits))


        return splits



    @staticmethod
    def __evaluate(id: LId,
                   features: np.ndarray, labels: np.ndarray,
                   idx_train: np.ndarray, idx_test: np.ndarray,
                   model: ValidationModel, loss_fn: LossFunc) -> Tuple[LId, float, np.ndarray]:
        
        #Train model and get predictions
        predictions: np.ndarray = model.train_predict(features[idx_train], labels[idx_train], 
                                                      features[idx_test])

        #Calculate loss and return it
        return (id, loss_fn(predictions, labels[idx_test]), predictions)



    @staticmethod
    def __execute_tasks(tasks, n_workers, verbosity_level) -> List[Tuple[LId, float]]:
        #NOTE: joblib should automatically memmap the numpy arrays
        return Parallel(n_jobs=n_workers, batch_size="auto",
                        verbose=verbosity_level)(tasks)



    def cross_validate(self, 
                       features: np.ndarray, labels: np.ndarray, 
                       models: List[ValidationModel], loss_fn: LossFunc) -> ValidationResult:
        """
        When doing 2-layer cross validation:
            Losses from inner test folds (n_outer*n_inner, n_model),
            Generalization error of inner folds inside an outer fold (n_outer, n_model),
            Index of best model in each outer split/fold (n_outer),
            Predictions of model on inner test sets (n_samples * (n_out - 1), n_model, shape_label),
            Labels on inner test sets (n_samples * (n_out - 1), shape_label),
            Losses of best models on outer test set (n_outer),
            Generalization error for model selection process (float),
            Predictions of best model on outer test sets (n_samples, shape_label),
            Labels on outer test sets (n_samples, shape_label),
        When doing 1-layer cross validation
            Losses of each folds (1*n_inner, n_model),
            Generalization error folds (1, n_model),
            Index of best model (1) (one element float array),
            Predictions of model on inner test sets (n_samples * (n_out - 1), n_model, n_pred),
            Labels on inner test sets (n_samples * (n_out - 1), n_pred),
        """


        #If verbose, print finished inner
        if self.verbose:
            print(f"Preparing inner fold evaluation ({max(self.n_outer, 1) * self.n_inner * len(models)} tasks)...")


        #If data should be shuffled, shuffle deterministically
        if self.randomize_seed is not None:
            features, labels = shuffle(
                features, labels,
                random_state = self.randomize_seed
            )
        #Else, use data as is
        else:
            pass


        #Get data splits to work on
        n_dataset = len(labels)
        idx_splits = CrossValidator.__create_idx_splits(
            n_dataset, self.n_outer, self.n_inner
        )
        
        
        #Create tasks for inner folds/splits
        tasks = []
        for i_o, (_, inners) in enumerate(idx_splits):
            for i_i, (idx_train, idx_test) in enumerate(inners):
                for i_m, model in enumerate(models):
                    tasks.append(delayed(CrossValidator.__evaluate)(
                        (i_o, i_i, i_m),
                        features, labels,
                        idx_train, idx_test, 
                        model, loss_fn
                    ))


        #Set verbosity level of processing,
        # =0 is no messages, 
        # >10 is all
        # >50 goes to stdout
        verbosity_level = 5 if self.verbose else 0
        
        #If verbose, print evaluating inner
        if self.verbose:
            print("Executing inner fold evaluation...")

        #Concurrently evaluate all inner splits
        results = CrossValidator.__execute_tasks(tasks, 
                                                 self.n_workers, 
                                                 verbosity_level)

        #If verbose, print finished inner
        if self.verbose:
            print("Finished evaluating inner folds.")
            print("Calculating inner fold statistics...")


        #Create array to store test/generalization errors
        #NOTE: Max of n_outer and 1 to account for one-layer scenario
        losses = np.empty((max(self.n_outer, 1),
                           self.n_inner,
                           len(models)),
                          dtype=np.float)
        preds_inner = [[[None
                         for _ in range(len(models))] 
                        for _ in range(self.n_inner)] 
                       for _ in range(max(self.n_outer, 1))]
        labels_inner = []


        #Store losses and predictions in array
        for id, loss, pred in results:
            (i_o, i_i, i_m) = id
            losses[id] = loss
            preds_inner[i_o][i_i][i_m] = pred
            
        #Store labels in array
        for i_o, (_, inners) in enumerate(idx_splits):
            for i_i, (_, idx_test) in enumerate(inners):
                for l in labels[idx_test]:
                    labels_inner.append(l)


        #Create test error array with columns for each model
        n_outer, n_inner, n_models = losses.shape
        inner_losses = losses.reshape((n_outer * n_inner, n_models))

        #Create predictions array
        inner_preds = [[] for _ in range(len(models))] 
        for o in preds_inner:
            for i in o:
                for i_m, m in enumerate(i):
                    for p in m:
                        inner_preds[i_m].append(p)
        inner_preds = np.array(inner_preds).swapaxes(0, 1)

        #Createe labels array
        inner_labels = np.array(labels_inner)


        #Estimate inner generalization error
        # for model 'x' for outer training split 'i' via
        # err_gen_x_i = sum(j, len(inner_j) / len(outer_i) * inner_i_j_x_loss)

        #Get (n_outer X n_inner X 1) matrix of fractions
        # denoting size of inner test split, relative to outer train split
        inner_test_fraction = np.expand_dims(np.array(
            [[len(i_test) / len(o_train) for _, i_test in inners]
             for (o_train, _), inners in idx_splits]
        ), 2)
        
        loss_gen_inner = (inner_test_fraction * losses).sum(axis = 1)
        
        idx_best_model = loss_gen_inner.argmin(axis = 1)


        #If verbose, print finished inner
        if self.verbose:
            print("Finished processing inner folds.")



        #If one-layer cross-validation, done, return
        if self.n_outer == 0:
            return ValidationResult(inner_losses, loss_gen_inner, idx_best_model,
                                    inner_preds, inner_labels)



        #If verbose, print preparing outer
        #NOTE: n_outer =/= 0 because else it would have returned
        if self.verbose:
            print(f"Preparing outer fold evaluation ({self.n_outer} tasks)...")


        #Create tasks for outer folds/splits
        tasks = []
        for idx_outer, (outer, _) in enumerate(idx_splits):
            idx_model = idx_best_model[idx_outer]
            idx_train, idx_test = outer

            tasks.append(delayed(CrossValidator.__evaluate)(
                idx_outer,
                features, labels,
                idx_train, idx_test, 
                models[idx_model], loss_fn
            ))


        #If verbose, print evaluating outer
        if self.verbose:
            print("Executing outer fold evaluation...")

        #Concurrently evaluate all outer splits
        results = CrossValidator.__execute_tasks(tasks, 
                                                 self.n_workers, 
                                                 verbosity_level)

        #If verbose, print finished outer
        if self.verbose:
            print("Finished evaluating outer folds.")
            print("Calculating outer fold statistics...")


        #Create array to store generalization errors
        loss_best_outer = np.empty(len(results), dtype=np.float)
        #Create arrays for predictions and labels
        preds_outer = [None for _ in range(len(results))]
        labels_outer = []

        #Store losses and predictions in array
        for id, loss, preds in results:
            loss_best_outer[id] = loss
            preds_outer[id] = preds

        #Store labels in array
        for idx_outer, (outer, _) in enumerate(idx_splits):
            _, idx_test = outer
            for l in labels[idx_test]:
                labels_outer.append(l)

        #Create predictions array
        outer_preds = [] 
        for o in preds_outer:
            for p in o:
                outer_preds.append(p)
        outer_preds = np.array(outer_preds)

        #Createe labels array
        outer_labels = np.array(labels_outer)


        #Get (n_outer) vector of fractions
        # denoting size of outer test split, relative to size of dataset
        outer_test_fraction = np.array(
            [len(o_test) / n_dataset for (_, o_test), _ in idx_splits]
        )
        #Estimate generalization error for model selection process via
        # err_gen = sum(len(outer_i) / len(dataset) * outer_i_loss)
        loss_gen_outer = (outer_test_fraction * loss_best_outer).sum()


        #If verbose, print finished inner
        if self.verbose:
            print("Finished processing outer folds.")


        #Return 2-layer cross-validation results
        return ValidationResult(inner_losses, loss_gen_inner, idx_best_model,
                                inner_preds, inner_labels, 
                                loss_best_outer, loss_gen_outer,
                                outer_preds, outer_labels)

