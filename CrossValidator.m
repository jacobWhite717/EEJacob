classdef CrossValidator
    %CrossValidator: namespace class for CV funcs
    properties
        run_parallel
    end
    
    methods
        function obj = CrossValidator(parallel_flag)
            arguments
                parallel_flag = false; 
            end
            obj.run_parallel = parallel_flag;
        end
    end
    
    methods (Static = true, Access = private)
        function folds = kfold_partitions(features, num_folds)
            arguments
                features TrialContainer
                num_folds double
            end

            [features, labels] = features.toMatrix();
            partitions = cvpartition(labels, 'kfold', num_folds, 'Stratify', true);

            for fold = 1:num_folds
                train_inds(:,fold) = training(partitions, fold);
                test_inds(:,fold) = test(partitions, fold);
            end

            folds = cell(size(train_inds, 2), 1);

            for i = 1:num_folds
                train_ind = train_inds(:,i);
                test_ind = test_inds(:,i);

                dataset.x_train = features(train_ind, :);
                dataset.x_test = features(test_ind, :);
                dataset.y_train = labels(train_ind, :);
                dataset.y_test = labels(test_ind, :);
                folds{i,1} = dataset;
            end
        end
        
        function blocks = block_partitions(features, num_blocks)
            arguments
                features TrialContainer
                num_blocks double
            end

            labels = features.getLabelsOfTrials();
            partitions = cvpartition(labels, 'kfold', num_blocks, 'Stratify', true);

            for fold = 1:num_blocks
                train_inds(:,fold) = training(partitions, fold);
                test_inds(:,fold) = test(partitions, fold);
            end  

            blocks = cell(size(train_inds, 2), 1);

            for i = 1:size(train_inds, 2)
                train_ind = train_inds(:,i);
                test_ind = test_inds(:,i);

                [dataset.x_train, dataset.y_train] = features.getTrialsByInd(train_ind).toMatrix();
                [dataset.x_test, dataset.y_test] = features.getTrialsByInd(test_ind).toMatrix();
                blocks{i,1} = dataset;
            end 
        end
        
        function blocks = random_block_partitions(features, num_blocks)
            arguments
                features TrialContainer
                num_blocks double
            end

            indexes = 1:features.numTrials();
            partitions = cvpartition(indexes, 'kfold', num_blocks);

            for fold = 1:num_blocks
                train_inds(:,fold) = training(partitions, fold);
                test_inds(:,fold) = test(partitions, fold);
            end  

            blocks = cell(size(train_inds, 2), 1);

            for i = 1:size(train_inds, 2)
                train_ind = train_inds(:,i);
                test_ind = test_inds(:,i);

                [dataset.x_train, dataset.y_train] = features.getTrialsByInd(train_ind).toMatrix();
                [dataset.x_test, dataset.y_test] = features.getTrialsByInd(test_ind).toMatrix();
                blocks{i,1} = dataset;
            end 
        end
        
        % used for fully_random_block_classify exclusively, shouldnt be used normally
        % results: ResultsContainer
        function [results, varargout] = random_block(obj, classifier_func, features, num_blocks, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_blocks {mustBeInteger,mustBeGreaterThan(num_blocks,1)}
                name_val_args.num_filtered_features double = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end
            num_filtered_features = name_val_args.num_filtered_features;
            runs = name_val_args.runs;
            save_classifiers = name_val_args.save_classifiers;
            clear name_val_args

            if save_classifiers
                classifiers = cell(runs*num_blocks, 1);
            end
            for run = 1:runs
                blocks = obj.random_block_partitions(features, num_blocks);
                
                temp_accuracy = zeros(num_blocks,1);
                temp_sensitivity  = zeros(num_blocks,1);
                temp_specificity = zeros(num_blocks,1);
                for i = 1:num_blocks
                    train_data = blocks{i}.x_train;
                    test_data = blocks{i}.x_test;
                    train_labels = blocks{i}.y_train;
                    test_labels = blocks{i}.y_test;

                    if strcmp(num_filtered_features, "all") || num_filtered_features == features.featureVectorLength()
                        % do nothing
                    elseif num_filtered_features ~= features.featureVectorLength()
                        [train_data, test_data] = obj.get_best_features(train_data, train_labels, num_filtered_features, test_features=test_data);
                    end

                    train_data = normalize(train_data);
                    test_data = normalize(test_data);

                    trained_model = classifier_func(train_data, train_labels, 'OptimizeHyperparameters', 'auto',...
                        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus', ...
                        'UseParallel', obj.run_parallel, 'ShowPlots', false, 'Verbose', 0));
                    if save_classifiers
                        classifiers{((run-1)*num_blocks)+i,1} = trained_model;
                    end
                    [accuracy, sensitivity, specificity] = obj.evaluate_model(trained_model, test_data, test_labels);
                    temp_accuracy(i,1) = accuracy;
                    temp_sensitivity(i,1) = sensitivity;
                    temp_specificity(i,1) = specificity;
                end
                results(run) = ResultsContainer(temp_accuracy, temp_sensitivity, temp_specificity);
                if save_classifiers
                    varargout{1} = classifiers;
                end
            end
        end

        function [opt_features, varargout] = get_best_features(features, labels, num_feats, name_val_args)
            arguments
                features (:,:) double
                labels (:,1) double
                num_feats double
                name_val_args.test_features (:,:) double = []
                name_val_args.scoring_func = @fscchi2
            end
            test_features = name_val_args.test_features;
            scoring_func = name_val_args.scoring_func;

            %fscmrmr, fscchi2
            idx = scoring_func(features, labels);
            opt_features = features(:, idx(1:num_feats));

            if ~isempty(test_features)
                opt_test_features = test_features(:, idx(1:num_feats));
                varargout{1} = opt_test_features;
            else
                varargout{1} = idx;
            end

        end

        function [accuracy, sensitivity, specificity, precision, recall, f1] = evaluate_model(classifier, test_data, test_labels)
            predictions = predict(classifier, test_data);
            
            conf_mat = confusionmat(test_labels, predictions);
            TN = conf_mat(1,1);
            FP = conf_mat(1,2);
            FN = conf_mat(2,1);
            TP = conf_mat(2,2);
        
            accuracy = (TP+TN)/(TP+TN+FP+FN)*100;
            sensitivity = (TP)/(TP+FN)*100;
            specificity = (TN)/(TN+FP)*100;

            precision = TP/(TP+FP)*100;
            recall = TP/(TP+FN)*100;
            f1 = 2*(precision*recall)/(precision+recall);
        end
    end
    
    methods (Access = public)
        % results: ResultsContainer
        function [results, varargout] = kfold(obj, classifier_func, features, num_folds, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_folds {mustBeInteger,mustBeGreaterThan(num_folds,1)}
                name_val_args.num_filtered_features = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end
            num_filtered_features = name_val_args.num_filtered_features;
            runs = name_val_args.runs;
            save_classifiers = name_val_args.save_classifiers;
            clear name_val_args

            if save_classifiers
                classifiers = cell(runs*num_folds, 1);
            end
            for run = 1:runs
                folds = obj.kfold_partitions(features, num_folds);

                temp_accuracy = zeros(num_folds,1);
                temp_sensitivity = zeros(num_folds,1);
                temp_specificity = zeros(num_folds,1);
                temp_precision = zeros(num_folds,1);
                temp_recall = zeros(num_folds,1);
                temp_f1 = zeros(num_folds,1);

                for i = 1:num_folds
                    train_data = folds{i}.x_train;
                    test_data = folds{i}.x_test;
                    train_labels = folds{i}.y_train;
                    test_labels = folds{i}.y_test;

                    if strcmp(num_filtered_features, "all") || num_filtered_features == features.featureVectorLength()
                        % do nothing
                    elseif num_filtered_features ~= features.featureVectorLength()
                        [train_data, test_data] = obj.get_best_features(train_data, train_labels, num_filtered_features, test_features=test_data);
                    end

                    train_data = normalize(train_data);
                    test_data = normalize(test_data);

                    trained_model = classifier_func(train_data, train_labels, 'OptimizeHyperparameters', 'auto',...
                        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus', ...
                        'UseParallel', obj.run_parallel, 'ShowPlots', false, 'Verbose', 0));
                    if save_classifiers
                        classifiers{((run-1)*num_folds)+i,1} = trained_model;
                    end
                    [accuracy, sensitivity, specificity, precision, recall, f1] = obj.evaluate_model(trained_model, test_data, test_labels);
                    temp_accuracy(i,1) = accuracy;
                    temp_sensitivity(i,1) = sensitivity;
                    temp_specificity(i,1) = specificity;

                    temp_precision(i,1) = precision;
                    temp_recall(i,1) = recall;
                    temp_f1(i,1) = f1;
                end
                results(run) = ResultsContainer(temp_accuracy, temp_sensitivity, temp_specificity, temp_precision, temp_recall, temp_f1);
                if save_classifiers
                    varargout{1} = classifiers;
                end
            end
        end

        % results: ResultsContainer
        function [results, varargout] = block(obj, classifier_func, features, num_blocks, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_blocks {mustBeInteger,mustBeGreaterThan(num_blocks,1)}
                name_val_args.num_filtered_features = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end
            num_filtered_features = name_val_args.num_filtered_features;
            runs = name_val_args.runs;
            save_classifiers = name_val_args.save_classifiers;
            clear name_val_args

            if save_classifiers
                classifiers = cell(runs*num_blocks, 1);
            end
            for run = 1:runs
                blocks = obj.block_partitions(features, num_blocks);
                
                temp_accuracy = zeros(num_blocks,1);
                temp_sensitivity  = zeros(num_blocks,1);
                temp_specificity = zeros(num_blocks,1);
                temp_precision = zeros(num_blocks,1);
                temp_recall = zeros(num_blocks,1);
                temp_f1 = zeros(num_blocks,1);

                for i = 1:num_blocks
                    train_data = blocks{i}.x_train;
                    test_data = blocks{i}.x_test;
                    train_labels = blocks{i}.y_train;
                    test_labels = blocks{i}.y_test;

                    if strcmp(num_filtered_features, "all") || num_filtered_features == features.featureVectorLength()
                        % do nothing
                    elseif num_filtered_features ~= features.featureVectorLength()
                        [train_data, test_data] = obj.get_best_features(train_data, train_labels, num_filtered_features, test_features=test_data);
                    end   

                    train_data = normalize(train_data);
                    test_data = normalize(test_data);

                    trained_model = classifier_func(train_data, train_labels, 'OptimizeHyperparameters', 'auto',...
                        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus', ...
                        'UseParallel', obj.run_parallel, 'ShowPlots', false, 'Verbose', 0));
                    if save_classifiers
                        classifiers{((run-1)*num_blocks)+i,1} = trained_model;
                    end
                    [accuracy, sensitivity, specificity, precision, recall, f1] = obj.evaluate_model(trained_model, test_data, test_labels);
                    temp_accuracy(i,1) = accuracy;
                    temp_sensitivity(i,1) = sensitivity;
                    temp_specificity(i,1) = specificity;

                    temp_precision(i,1) = precision;
                    temp_recall(i,1) = recall;
                    temp_f1(i,1) = f1;
                end
                results(run) = ResultsContainer(temp_accuracy, temp_sensitivity, temp_specificity, temp_precision, temp_recall, temp_f1);
                if save_classifiers
                    varargout{1} = classifiers;
                end
            end
        end

        function results = kfold_random(obj, classifier_func, features, num_folds, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_folds double
                name_val_args.num_filtered_features double = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end


            features = LabelRandomizer.FullRandomize(features);
            results = obj.kfold(classifier_func, features, num_folds, ...
                num_filtered_features=name_val_args.num_filtered_features, ...
                runs=name_val_args.runs, ...
                save_classifiers=name_val_args.save_classifiers);
        end

        function results = block_random(obj, classifier_func, features, num_folds, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_folds {mustBeInteger,mustBeGreaterThan(num_folds,1)}
                name_val_args.num_filtered_features double = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end

            features = LabelRandomizer.BlockRandomize(features);
            results = obj.block(classifier_func, features, num_folds, ...
                num_filtered_features=name_val_args.num_filtered_features, ...
                runs=name_val_args.runs, ...
                save_classifiers=name_val_args.save_classifiers);
        end

        function results = block_random_kfold_classify(obj, classifier_func, features, num_folds, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_folds {mustBeInteger,mustBeGreaterThan(num_folds,1)}
                name_val_args.num_filtered_features double = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end

            features = LabelRandomizer.BlockRandomize(features);
            results = obj.kfold(classifier_func, features, num_folds, ...
                num_filtered_features=name_val_args.num_filtered_features, ...
                runs=name_val_args.runs, ...
                save_classifiers=name_val_args.save_classifiers);
        end
        
        function results = fully_random_block_classify(obj, classifier_func, features, num_folds, name_val_args)
            arguments
                obj
                classifier_func 
                features TrialContainer
                num_folds {mustBeInteger,mustBeGreaterThan(num_folds,1)}
                name_val_args.num_filtered_features double = features.featureVectorLength()
                name_val_args.runs {mustBeInteger,mustBeNonnegative} = 1
                name_val_args.save_classifiers = false
            end

            features = LabelRandomizer.FullRandomize(features);
            results = obj.random_block(classifier_func, features, num_folds, ...
                num_filtered_features=name_val_args.num_filtered_features, ...
                runs=name_val_args.runs, ...
                save_classifiers=name_val_args.save_classifiers);
        end
        
    end 
end

