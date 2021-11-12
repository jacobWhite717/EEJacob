labels_folder = "labels_classes_percentile/";
num_trials = 40;
num_subjects = 32;
participant_pool = (1:num_subjects);

for s = participant_pool
    %% load datasets into EEGLAB and labels
    fprintf('Working on subject %i\n', s);
    disp('Loading subject dataset...');
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    temp_cell = {};
    parfor task = 1:num_trials
        file_path = ['E:\\MUN\\Masters\\EEG CV Working\\DEAP CV tests\\data_preprocessed_eeg\\' sprintf('s%02i', s), '\\'];
        file_name = sprintf('s%02i_t%02i.set', s, task);
        EEG = pop_loadset('filename', file_name, 'filepath', file_path);
        temp_cell{task} = EEG;  % store EEG in a temporary cell for parfor compatibility
    end
    for task = 1:num_trials
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, temp_cell{task});  % update EEGLAB vars outside of parfor loop, as they will be needed afterwards, and vars made within parfor loops are lost
    end
    trials_labels = load(sprintf('%ss%02i_labels.mat', labels_folder, s)).labels_class;

    %% get DE features
    disp('Calculating features...');
    trials_features = TrialContainer(); % container for all combined and calculated feature sets, however only using DE here
    
    temp_cell = {};
    parfor i = 1:length(ALLEEG) 
        diff_entrops = GetDifferentialEntropy(ALLEEG(i));
        de_feat_epochs = Epoch.FromMatrix(diff_entrops, label=trials_labels(i));
        temp_cell{i} = de_feat_epochs;  % same idea again with the temp_cell
    end
    de_features = TrialContainer(trials=temp_cell);
    trials_features = trials_features.combineTrialContainerEpochs(de_features);

    save(sprintf('PreparedFeatures/DE/s%02i.mat', s), 'trials_features')
end


%% helper funcs
function [delta, theta, alpha, beta, gamma] = GetSubBands (EEG)
    arguments
        EEG
    end
    delta = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 3);
    theta = pop_eegfiltnew(EEG, 'locutoff', 4, 'hicutoff', 7);
    alpha = pop_eegfiltnew(EEG, 'locutoff', 8, 'hicutoff', 13);
    beta  = pop_eegfiltnew(EEG, 'locutoff', 14, 'hicutoff', 30);
    gamma = pop_eegfiltnew(EEG, 'locutoff', 31, 'hicutoff', 50);
end


% takes a single trials data, i.e. run in loop for all trials
% feats = epochs/features (r/c) 
function feats = GetDifferentialEntropy(EEG)
    arguments
        EEG
    end
    num_electrodes = size(EEG.data, 1);
    num_epochs = size(EEG.data, 3);
    feats = [];
    [delta, theta, alpha, beta, gamma] = GetSubBands(EEG);
    for electrode = 1:num_electrodes
        electrode_feats = [];
        delta_electrode_data = cast(squeeze(delta.data(electrode,:,:)), 'double');
        theta_electrode_data = cast(squeeze(theta.data(electrode,:,:)), 'double');
        alpha_electrode_data = cast(squeeze(alpha.data(electrode,:,:)), 'double');
        beta_electrode_data  = cast(squeeze(beta.data(electrode,:,:)),  'double');
        gamma_electrode_data = cast(squeeze(gamma.data(electrode,:,:)), 'double');
        for epoch = 1:num_epochs
            delta_DE = estimateEntropyUsingCopulas(delta_electrode_data(:,epoch));
            theta_DE = estimateEntropyUsingCopulas(theta_electrode_data(:,epoch));
            alpha_DE = estimateEntropyUsingCopulas(alpha_electrode_data(:,epoch));
            beta_DE  = estimateEntropyUsingCopulas(beta_electrode_data(:,epoch));
            gamma_DE = estimateEntropyUsingCopulas(gamma_electrode_data(:,epoch));
            electrode_feats = [electrode_feats; delta_DE, theta_DE, alpha_DE, beta_DE, gamma_DE];
        end
        feats = [feats, electrode_feats];
    end
end