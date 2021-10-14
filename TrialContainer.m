classdef TrialContainer
    
    properties (SetAccess=private)
        trials cell  % cell of Epoch arrays
        same_class_trials
    end
    
    methods (Access = private)
        function obj = verify_same_class_trials(obj)
            if obj.numTrials() == 0
                obj.same_class_trials = false;
            else
                bool = true;
                for i = 1:obj.numTrials()
                    trial = obj.trials{i};
                    if trial.verifySameClass() == -1
                        bool = false;                    
                    end
                end
                obj.same_class_trials = bool;
            end
        end
        
    end
    
    methods
        
        function obj = TrialContainer(NameValueArgs)
            % ctor
            arguments
                NameValueArgs.trials (:,1) cell = {}
            end
            obj.trials = NameValueArgs.trials;
            obj = obj.verify_same_class_trials();
        end
        
        function num = numTrials(obj)
            % {return}: (double) length of trials
            num = length(obj.trials);
        end
        
        function num = featureVectorLength(obj)
            if obj.numTrials ~= 0
                num = length(obj.trials{1}(1).features);
            else
                error("Feature vector not yet defined")
            end
        end
        
        function obj = addTrialFromEpochs(obj, epochs)
            % {return}: (TrialContainer) new updated object
            % {arg} obj: (TrialConatiner) object
            % {arg} epochs: (Epoch) to add to a new trial
            arguments
                obj 
                epochs (:,1) Epoch 
            end
            
            obj.trials{end+1,1} = epochs;
            obj = obj.verify_same_class_trials();
        end
        
        function obj = combineTrialContainerEpochs(obj, other)
            % {return}: new object with epochs containing features from both TrialContainers
            % {arg} obj: (TrialContainer) object
            % {arg} other: (TrialContainer) object to be appended to obj
            arguments
                obj TrialContainer
                other TrialContainer
            end
            
            if ( obj.numTrials() == 0 && other.numTrials() ~= 0 )  % adding to an empty container
                obj.trials = other.trials;
            elseif ( obj.numTrials() == other.numTrials() )
                for i = 1:other.numTrials()
                    if length(obj.trials{i,1}) == length(other.trials{i,1})
                        obj_cur = obj.trials{i,1};
                        oth_cur = other.trials{i,1};
                        for j = 1:length(obj_cur)
                            obj_cur(j,1) = obj_cur(j,1).appendEpoch(oth_cur(j,1));
                        end
                        obj.trials{i,1} = obj_cur;
                    else
                        error("Number of epochs in trials not equal")
                    end
                end
            else 
                error("Number of trials in containers not equal")
            end
            
            obj = obj.verify_same_class_trials();
        end
        
        function new_container = appendTrialContainer(obj, other)
            % {return} new object containing trials stacked from both TrialContainers
            arguments
                obj TrialContainer
                other TrialContainer
            end

            new_container = TrialContainer();
            new_container.trials = [obj.trials; other.trials];
            
            new_container = new_container.verify_same_class_trials();
        end

        function new_cont = getTrialsByInd(obj, ind)
            % {return}: (TrialContainer) new TrialConatiner with trials from obj, of indicies ind
            % {arg} ind: (double array), indicies of obj.trials to select
            wanted = obj.trials(ind);
            new_cont = TrialContainer();
            new_cont.trials = wanted;
            
            new_cont = new_cont.verify_same_class_trials();
        end
        
        function new_cont = getTrialsByClass(obj, class)
            % {return}: (TrialContainer) new TrialConatiner, consisting of trials belonging to class/label 'class'
            % {arg} class: (double) of the desired class to gather from
            arguments
                obj
                class (1,1) double
            end
                        
            class_inds = [];
            for i = 1:obj.numTrials()
                if obj.trials{i}.verifySameClass() == class
                    class_inds = [class_inds; i];
                end
            end
            new_cont = obj.getTrialsByInd(class_inds);
            new_cont = new_cont.verify_same_class_trials();
        end
        
        % fixme does this really need to throw an error if it already has the validation?
        function labels = getLabelsOfTrials(obj)
            if obj.same_class_trials
                labels = [];
                for i = 1:obj.numTrials()
                    if obj.trials{i}.verifySameClass() ~= -1 
                        labels = [labels; obj.trials{i}.verifySameClass()];
                    else 
                        error("Trial not all of the same epoch")
                    end
                end
            else
                labels = -1;
            end

        end
        
        function [feats, labels] = toMatrix(obj)
            % {return} feats: a (double matrix) of epochs/features from all trials 
            % {return} labels: a (double array) of the respective class ofeach epoch
            num_trials = obj.numTrials();
            feats = [];
            labels = [];
            for i = 1:num_trials
                [cur_trial, cur_label] = obj.trials{i}.toMatrix();
                feats = [feats; cur_trial];
                labels = [labels; cur_label];
            end
        end

    end
end

