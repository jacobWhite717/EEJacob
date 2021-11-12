%% Jacob White (jrw111@mun.ca) DEAP DB tests
%% global variables and conatiners 
t_start = tic;
optimize_single_run = false;
num_trials = 40;
num_subjects = 32;
verbose = false;

% labels_folder = "labels_classes/";
labels_folder = "labels_classes_percentile/";

results_lda_kfold_table = cell(num_subjects, 1);
results_lda_block_table = cell(num_subjects, 1);

results_knn_kfold_table = cell(num_subjects, 1);
results_knn_block_table = cell(num_subjects, 1);


participant_pool = (1:num_subjects);


%%
parfor s = participant_pool
    %% load pre-saved features in TrialContainer
    fprintf('Working on subject %i\n', s);
    load_file_name = sprintf('PreparedFeatures/DE/s%02i.mat', s);
    trials_features = load(load_file_name).trials_features;
    
    
    %% separate out features by class
    trials_pos = trials_features.getTrialsByClass(2);
    % trials_neu = trials_features.getTrialsByClass(1);
    trials_neg = trials_features.getTrialsByClass(0);

    trials_pos_neg = trials_pos.appendTrialContainer(trials_neg);
    
    
    %% classification params
    cv = CrossValidator(optimize_single_run);
    folds = 5;
    runs = 5;

    
    %% k-fold classification
    disp('Performing k-fold classification...')
    % lda
    classifier_func = @fitcdiscr; 
    results_lda_kfold = cv.kfold(classifier_func, trials_pos_neg, folds, ...
        num_filtered_features=round(trials_pos_neg.featureVectorLength()/2), ...
        runs=runs);
    results_lda_kfold_table{s} = results_lda_kfold.makeResultsTable();
    
    % knn
    classifier_func = @fitcknn; 
    results_knn_kfold = cv.kfold(classifier_func, trials_pos_neg, folds, ...
        num_filtered_features=round(trials_pos_neg.featureVectorLength()/2), ...
        runs=runs);
    results_knn_kfold_table{s} = results_knn_kfold.makeResultsTable();

    
    %% blocked CV
    disp('Performing blocked classification...')
    % lda
    classifier_func = @fitcdiscr;
    results_lda_block = cv.block(classifier_func, trials_pos_neg, folds, ...
        num_filtered_features=round(trials_pos_neg.featureVectorLength()/2), ...
        runs=runs);
    results_lda_block_table{s} = results_lda_block.makeResultsTable();
    
    % knn
    classifier_func = @fitcknn;
    results_knn_block = cv.block(classifier_func, trials_pos_neg, folds, ...
        num_filtered_features=round(trials_pos_neg.featureVectorLength()/2), ...
        runs=runs);
    results_knn_block_table{s} = results_knn_block.makeResultsTable();
end


%% results
% file_name = 'temp';
file_name = 'nov5 DEAP 1 class split into 2 and classified';
excel_name = ['results analysis/', file_name,'.xlsx'];
mat_name = ['results analysis/', file_name, '.mat'];
save(mat_name);

summary_lda_kfold = make_results_summary(results_lda_kfold_table);
summary_lda_block = make_results_summary(results_lda_block_table);
summary_knn_kfold = make_results_summary(results_knn_kfold_table);
summary_knn_block = make_results_summary(results_knn_block_table);

sheet_name = "Summary";
writematrix("LDA classifier", excel_name, 'Sheet', sheet_name, 'Range', 'B2');
writematrix("kfold - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'B3');
writetable(summary_lda_kfold, excel_name, 'Sheet', sheet_name, 'Range', 'B4', 'WriteRowNames', true);

writematrix("LDA classifier", excel_name, 'Sheet', sheet_name, 'Range', 'G2');
writematrix("block - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'G3');
writetable(summary_lda_block, excel_name, 'Sheet', sheet_name, 'Range', 'G4', 'WriteRowNames', true);

writematrix("KNN classifier", excel_name, 'Sheet', sheet_name, 'Range', 'L2');
writematrix("kfold - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'L3');
writetable(summary_knn_kfold, excel_name, 'Sheet', sheet_name, 'Range', 'L4', 'WriteRowNames', true);

writematrix("KNN classifier", excel_name, 'Sheet', sheet_name, 'Range', 'Q2');
writematrix("block - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'Q3');
writetable(summary_knn_block, excel_name, 'Sheet', sheet_name, 'Range', 'Q4', 'WriteRowNames', true);

for i = participant_pool
    sheet_name = sprintf('Sub %02i', i);
    writematrix("LDA classifier", excel_name, 'Sheet', sheet_name, 'Range', 'B2');
    writematrix("kfold - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'B3');
    writetable(results_lda_kfold_table{i}, excel_name, 'Sheet', sheet_name, 'Range', 'B4', 'WriteRowNames', true);
    
    writematrix("KNN classifier", excel_name, 'Sheet', sheet_name, 'Range', 'B13');
    writematrix("kfold - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'B14');
    writetable(results_knn_kfold_table{i}, excel_name, 'Sheet', sheet_name, 'Range', 'B15', 'WriteRowNames', true);
    
    writematrix("block - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'G3');
    writetable(results_lda_block_table{i}, excel_name, 'Sheet', sheet_name, 'Range', 'G4', 'WriteRowNames', true);
    
    writematrix("block - Positive/Negative Classification", excel_name, 'Sheet', sheet_name, 'Range', 'G14');
    writetable(results_knn_block_table{i}, excel_name, 'Sheet', sheet_name, 'Range', 'G15', 'WriteRowNames', true);
end

run_time = toc(t_start);
sprintf("Total run time was %.3fs", run_time)


%% Helper functions
function summary = make_results_summary(results)
    summary = zeros(length(results)+2,3);
    for i = 1:length(results)
        summary(i,1) = results{i}.Accuracy(end-1);
        summary(i,2) = results{i}.Sensitivity(end-1);
        summary(i,3) = results{i}.Specificity(end-1);
    end
    summary(end-1,1) = mean(summary(1:end-2,1));
    summary(end-1,2) = mean(summary(1:end-2,2));
    summary(end-1,3) = mean(summary(1:end-2,3));

    summary(end,1) = std(summary(1:end-2,1));
    summary(end,2) = std(summary(1:end-2,2));
    summary(end,3) = std(summary(1:end-2,3));
    
    col_names = [ "Accuracy", "Sensitivity", "Specificity" ];
    row_names = [];
    for i = 1:length(results)
        row_names = [row_names, string(sprintf('Sub %i', i))];
    end
    row_names = [row_names, "mean", "std"];
    summary = array2table(summary, 'RowNames', row_names, 'VariableNames', col_names);
end


function split_class_trials = split_single_class_trials(trial_cont)
    arguments
        trial_cont TrialContainer
    end

    if trial_cont.getLabelsOfTrials() == -1
        error("Trials are not uniformly classed") ;
    end

    if mod(length(trial_cont.trials), 2)
        error("Odd number of trials");
    end

    inds_perm1 = randperm(trial_cont.numTrials());
    for i = inds_perm1(1:ceil(length(inds_perm1)/2))
        swapped_trial = trial_cont.trials{i};
        for j = 1:length(swapped_trial)
            swapped_trial(j).class = 1;
        end
        split_class_trials{i} = swapped_trial;
    end
    for i = inds_perm1( ceil(length(inds_perm1)/2)+1:length(inds_perm1) )
        swapped_trial = trial_cont.trials{i};
        for j = 1:length(swapped_trial)
            swapped_trial(j).class = 2;
        end
        split_class_trials{i} = swapped_trial;
    end
    split_class_trials = TrialContainer(trials=split_class_trials);
end
