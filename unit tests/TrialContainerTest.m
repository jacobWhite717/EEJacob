classdef TrialContainerTest < matlab.unittest.TestCase
    methods (Test)
        function TestCombineEpochs(testCase)
            epoch1(1) = Epoch([1, 1, 1], 1);
            epoch1(2) = Epoch([1, 1, 1], 2);

            epoch2(1) = Epoch([2, 2, 2], 1);
            epoch2(2) = Epoch([2, 2, 2], 2);

            t_cont1 = TrialContainer();
            t_cont1 = t_cont1.addTrialFromEpochs(epoch1);
            t_cont1 = t_cont1.addTrialFromEpochs(epoch2);

            t_cont2 = TrialContainer();
            t_cont2 = t_cont2.addTrialFromEpochs(epoch1);
            t_cont2 = t_cont2.addTrialFromEpochs(epoch2);

            t_cont3 = t_cont1.combineTrialContainerEpochs(t_cont2);

            testCase.verifyEqual( t_cont3.trials{1}(1), epoch1(1).appendEpoch(epoch1(1)) );
            testCase.verifyEqual( t_cont3.trials{1}(2), epoch1(2).appendEpoch(epoch1(2)) );
            testCase.verifyEqual( t_cont3.trials{2}(1), epoch2(1).appendEpoch(epoch2(1)) );
            testCase.verifyEqual( t_cont3.trials{2}(2), epoch2(2).appendEpoch(epoch2(2)) );
    
        end
        
        function TestGettingTrialsByInd(testCase)
            epoch1 = Epoch(1, 1);
            epoch2 = Epoch(2, 2);
            epoch3 = Epoch(3, 3);
            epoch4 = Epoch(4, 4);

            t_cont = TrialContainer();
            t_cont = t_cont.addTrialFromEpochs(epoch1);
            t_cont = t_cont.addTrialFromEpochs(epoch2);
            t_cont = t_cont.addTrialFromEpochs(epoch3);
            t_cont = t_cont.addTrialFromEpochs(epoch4);
            
            sliced_t = t_cont.getTrialsByInd(2:3);
            
            testCase.verifyEqual( sliced_t.trials{1}, epoch2 );
            testCase.verifyEqual( sliced_t.trials{2}, epoch3 );
        end
        
        function TestGettingTrialsByClass(testCase)
            epochs_t1 = Epoch.FromMatrix(ones(5,2), 1);
            epochs_t2 = Epoch.FromMatrix(2*ones(5,2), 2);
            epochs_t3 = Epoch.FromMatrix(3*ones(5,2), 3);
            
            tcont = TrialContainer();
            tcont = tcont.addTrialFromEpochs(epochs_t1);
            tcont = tcont.addTrialFromEpochs(epochs_t2);
            tcont = tcont.addTrialFromEpochs(epochs_t3);
            
            tcont_class2 = tcont.getTrialsByClass(2);
            
            testCase.verifyEqual( tcont_class2.numTrials(), 1 );
            testCase.verifyEqual( tcont_class2.trials{1}.toMatrix(), epochs_t2.toMatrix() );
        end
        
        function TestToFeatureVector(testCase)
            epochs_t1 = Epoch.FromMatrix(ones(5,2), 1);
            epochs_t2 = Epoch.FromMatrix(2*ones(5,2), 2);
            
            tcont = TrialContainer();
            tcont = tcont.addTrialFromEpochs(epochs_t1);
            tcont = tcont.addTrialFromEpochs(epochs_t2);
            
            [tmat, tlabs] = tcont.toMatrix();
            
            correct_mat = [ones(5,2); 2*ones(5,2)];
            correct_lab = [ones(5,1); 2*ones(5,1)];
            
            testCase.verifyEqual( tmat, correct_mat );
            testCase.verifyEqual( tlabs, correct_lab );
        end

        function TestAppendContainers(testCase)
            epochs_t1 = Epoch.FromMatrix(ones(5,2), 1);
            epochs_t2 = Epoch.FromMatrix(2*ones(5,2), 2);

            tcont1 = TrialContainer();
            tcont1 = tcont1.addTrialFromEpochs(epochs_t1);

            tcont2 = TrialContainer();
            tcont2 = tcont2.addTrialFromEpochs(epochs_t2);

            new_cont = tcont1.appendTrialContainer(tcont2);

            correct_result = [ones(5, 2); 2*ones(5, 2)];

            testCase.verifyEqual( new_cont.toMatrix(), correct_result);
        end
        
        function TestNameValueCtor(testCase)
            epoch1 = Epoch(1, 1);
            epoch2 = Epoch(2, 2);
            epoch3 = Epoch(3, 3);
            epoch4 = Epoch(4, 4);
            epochs = {epoch1; epoch2; epoch3; epoch4};
            
            tcont = TrialContainer(trials=epochs);
            
            correct_result = [1; 2; 3; 4];
            
            testCase.verifyEqual( tcont.toMatrix, correct_result );
            testCase.verifyEqual( tcont.same_class_trials, true );
        end
    end
end

