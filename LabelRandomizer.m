classdef LabelRandomizer
    % think of this as a namespace, because I dont know how to make one in matlab
    properties
    end
    
    methods
        function obj = LabelRandomizer()
            % ctor
        end
    end
    
    methods (Static)
        function swapped_class_trials = FullRandomize(trial_cont)
            arguments
                trial_cont TrialContainer 
            end

            
            [mat, lab] = trial_cont.toMatrix();
            specific_labels = unique(lab);
            
            class1_ind = find(lab == specific_labels(1));
            class1_ind_perm = class1_ind(randperm(length(class1_ind)));
            
            class2_ind = find(lab == specific_labels(2));
            class2_ind_perm = class2_ind(randperm(length(class2_ind)));
            
            for i = class1_ind_perm(1:ceil(length(class1_ind_perm)/2))
                lab(i) = specific_labels(2);
            end
            for i = class2_ind_perm(1:ceil(length(class2_ind_perm)/2))
                lab(i) = specific_labels(1);
            end
            
            epochs = Epoch.FromMatrix(mat, labels=lab);
            trial_length = length(trial_cont.trials{1});
            trials = {};
            for i = 1:trial_cont.numTrials()
                trials{i,1} = epochs((i-1)*trial_length+1:(i-1)*trial_length+trial_length,1);
            end
            
            swapped_class_trials = TrialContainer(trials=trials);
        end
        
        % rounds swapped trials count up, i.e. w/ 5 trails of a class, 3 are swapped
        function swapped_class_trials = BlockRandomize(trial_cont)
            arguments
                trial_cont TrialContainer
            end

            if trial_cont.getLabelsOfTrials() == -1
                error("Trials are not uniformly classed") ;
            end

            all_labels = trial_cont.getLabelsOfTrials();
            specific_labels = unique(all_labels);
            
            trials_class1 = trial_cont.getTrialsByClass(specific_labels(1));
            trials_class2 = trial_cont.getTrialsByClass(specific_labels(2));
            swapped_class1 = {};
            swapped_class2 = {};
            
            inds_perm1 = randperm(trials_class1.numTrials());
            for i = inds_perm1(1:ceil(length(inds_perm1)/2))
                swapped_trial = trials_class1.trials{i};
                for j = 1:length(swapped_trial)
                    swapped_trial(j).class = specific_labels(2);
                end
                swapped_class1{i} = swapped_trial;
            end
            for i = inds_perm1( ceil(length(inds_perm1)/2)+1:length(inds_perm1) )
                same_trial = trials_class1.trials{i};
                swapped_class1{i} = same_trial;
            end
            swapped_class1 = TrialContainer(trials=swapped_class1);


            inds_perm2 = randperm(trials_class2.numTrials());
            for i = inds_perm2(1:ceil(length(inds_perm2)/2))
                swapped_trial = trials_class2.trials{i};
                for j = 1:length(swapped_trial)
                    swapped_trial(j).class = specific_labels(1);
                end
                swapped_class2{i} = swapped_trial;
            end
            for i = inds_perm2( ceil(length(inds_perm2)/2)+1:length(inds_perm2) )
                same_trial = trials_class2.trials{i};
                swapped_class2{i} = same_trial;
            end
            swapped_class2 = TrialContainer(trials=swapped_class2);

            swapped_class_trials = swapped_class1.appendTrialContainer(swapped_class2);
        end
    end
end

