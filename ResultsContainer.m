classdef ResultsContainer
    
    properties
        accuracy
        sensitivity
        specificity

        mean_accuracy
        std_accuracy

        mean_sensitivity
        std_sensitivity
        
        mean_specificity
        std_specificity
    end
    
    methods
        function obj = ResultsContainer(accuracy, sensitivity, specificity)
            arguments
                accuracy (:,1) double
                sensitivity (:,1) double
                specificity (:,1) double
            end
            assert(length(accuracy)==length(sensitivity), "Input vector lengths not equal");
            assert(length(specificity)==length(sensitivity), "Input vector lengths not equal")

            obj.accuracy = accuracy;
            obj.sensitivity = sensitivity;
            obj.specificity = specificity;

            obj.mean_accuracy = mean(obj.accuracy);
            obj.std_accuracy = std(obj.accuracy);

            obj.mean_sensitivity = mean(obj.sensitivity);
            obj.std_sensitivity = std(obj.sensitivity);

            obj.mean_specificity = mean(obj.specificity);
            obj.std_specificity = std(obj.specificity);
        end
        
        function tbl = makeResultsTable(obj)
            num = length(obj);
            results_array = zeros(num, 3);
            for i = 1:num
                results_array(i, 1) = obj(i).mean_accuracy;
                results_array(i, 2) = obj(i).mean_sensitivity;
                results_array(i, 3) = obj(i).mean_specificity;
            end
            results_array = [results_array; ...
                mean(results_array(:,1)), mean(results_array(:,2)), mean(results_array(:,3)); ...
                std(results_array(:,1)), std(results_array(:,2)), std(results_array(:,3))];


            col_names = [ "Accuracy", "Sensitivity", "Specificity" ];
            row_names = [];
            for i = 1:num
                row_names = [row_names, string(sprintf('Run %i', i))];
            end
            row_names = [row_names, "mean", "std"];

            tbl = array2table(results_array, 'RowNames', row_names, 'VariableNames', col_names);
        end
        
    end
end

