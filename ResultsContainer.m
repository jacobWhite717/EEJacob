classdef ResultsContainer
    
    properties
        accuracy
        sensitivity
        specificity
        precision
        recall
        f1

        mean_accuracy
        std_accuracy

        mean_sensitivity
        std_sensitivity
        
        mean_specificity
        std_specificity
        
        mean_precision
        std_precision

        mean_recall
        std_recall

        mean_f1
        std_f1
    end
    
    methods
        function obj = ResultsContainer(accuracy, sensitivity, specificity, precision, recall, f1)
            arguments
                accuracy (:,1) double
                sensitivity (:,1) double
                specificity (:,1) double
                precision (:,1) double
                recall (:,1) double
                f1 (:,1) double
            end
            assert(length(accuracy)==length(sensitivity), "Input vector lengths not equal");
            assert(length(accuracy)==length(specificity), "Input vector lengths not equal");
            assert(length(accuracy)==length(precision), "Input vector lengths not equal");
            assert(length(accuracy)==length(recall), "Input vector lengths not equal");
            assert(length(accuracy)==length(f1), "Input vector lengths not equal");

            obj.accuracy = accuracy;
            obj.sensitivity = sensitivity;
            obj.specificity = specificity;
            obj.precision = precision;
            obj.recall = recall;
            obj.f1 = f1;

            %todo are these even used?
            obj.mean_accuracy = mean(obj.accuracy);
            obj.std_accuracy = std(obj.accuracy);

            obj.mean_sensitivity = mean(obj.sensitivity);
            obj.std_sensitivity = std(obj.sensitivity);

            obj.mean_specificity = mean(obj.specificity);
            obj.std_specificity = std(obj.specificity);

            obj.mean_precision = mean(obj.precision);
            obj.std_precision = std(obj.precision);

            obj.mean_recall = mean(obj.recall);
            obj.std_recall = std(obj.recall);

            obj.mean_f1 = mean(obj.f1);
            obj.std_f1 = std(obj.f1);
        end
        
        % creates a table from 1 or an array of multiple ResultsTable objects
        function tbl = makeResultsTable(obj)
            num_runs = length(obj);
            results_array = zeros(num_runs, 12);
            for i = 1:num_runs
                results_array(i, 1) = obj(i).mean_accuracy;
                results_array(i, 2) = obj(i).std_accuracy;
                results_array(i, 3) = obj(i).mean_sensitivity;
                results_array(i, 4) = obj(i).std_sensitivity;
                results_array(i, 5) = obj(i).mean_specificity;
                results_array(i, 6) = obj(i).std_specificity;
                results_array(i, 7) = obj(i).mean_precision;
                results_array(i, 8) = obj(i).std_precision;
                results_array(i, 9) = obj(i).mean_recall;
                results_array(i, 10) = obj(i).std_recall;
                results_array(i, 11) = obj(i).mean_f1;
                results_array(i, 12) = obj(i).std_f1;
            end
            results_array = [results_array; ...
                mean(results_array(:,1)), -1, mean(results_array(:,3)), -1, mean(results_array(:,5)), -1, mean(results_array(:,7)), -1, mean(results_array(:,9)), -1, mean(results_array(:,11)), -1; ...
                std(results_array(:,1)), -1, std(results_array(:,3)), -1, std(results_array(:,5)), -1, std(results_array(:,7)), -1, std(results_array(:,9)), -1, std(results_array(:,11)), -1];


            col_names = [ "Accuracy", "StdAcc", "Sensitivity", "StdSens", "Specificity", "StdSpec", ...
                "Precision", "StdPrec", "Recall", "StdRec", "F1", "StdF1"];
            row_names = [];
            for i = 1:num_runs
                row_names = [row_names, string(sprintf('Run %i', i))];
            end
            row_names = [row_names, "mean", "std"];

            tbl = array2table(results_array, 'RowNames', row_names, 'VariableNames', col_names);
        end
        
    end
end

