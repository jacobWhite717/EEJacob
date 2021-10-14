classdef Epoch
    
    properties % (SetAccess=private)
        features (1,:) double
        class double
    end
    
    methods
        % ctor
        function obj = Epoch(features, class)
            obj.features = features;
            obj.class = class;
        end
                
        function obj = addFeatures(obj, features)
            arguments
                obj Epoch
                features (1,:)
            end
            obj.features = [obj.features, features];
        end
        
        function obj = appendEpoch(obj, other)
            arguments
                obj Epoch
                other Epoch
            end
            if obj.class == other.class
                obj.features = [obj.features, other.features];
            else
                error("Epochs are not of the same class")
            end
        end
        
        % if all labels in the epochs are the same, returns the class label
        % if all labels are not the same, returns -1
        function class = verifySameClass(obj)
            num_epochs = length(obj);
            if num_epochs == 1
                class = obj.class;
            elseif num_epochs > 1
                [~, labels] = obj.toMatrix();
                if all(labels == labels(1))
                    class = labels(1);
                else
                    class = -1;
                end
            else
                class = -1;
            end
        end
        
        function [feats, labels] = toMatrix(obj)
            num_epochs = length(obj);
            if num_epochs == 1
                feats = obj.features;
                labels = obj.class;
            elseif num_epochs > 1 
                num_feats = length(obj(1).features);
                feats = zeros(num_epochs, num_feats);
                labels = zeros(num_epochs, 1);
                for i = 1:num_epochs
                    feats(i,:) = obj(i).features;
                    labels(i,1) = obj(i).class;
                end
            else % empty epoch somehow
                feats = [];
                labels = [];
            end
        end
        
        function bool = eq(obj, other)
            bool = all(obj.features == other.features);
            bool = bool && obj.class == other.class;
        end
    end
    
    methods (Static)
        % features should be a matrix of shape epochs/features
        % label is a label to give to all epochs generated       
        % labels is an array of labels for all epochs 
        % label and labels are mutually exclusive
        function epochs = FromMatrix(features, NameValueArgs)
            arguments
                features
                NameValueArgs.label (1,1) double 
                NameValueArgs.labels (:,1) double 
            end
            
            if isfield(NameValueArgs, 'label')
                for i = 1:size(features, 1)
                    cur_features = features(i,:);
                    cur_epoch = Epoch(cur_features, NameValueArgs.label);
                    epochs(i,1) = cur_epoch;
                end
            elseif isfield(NameValueArgs, 'labels')
                for i = 1:size(features, 1)
                    cur_features = features(i,:);
                    cur_label = NameValueArgs.labels(i);
                    cur_epoch = Epoch(cur_features, cur_label);
                    epochs(i,1) = cur_epoch;
                end
            else
                error("Must supply either label or labels argument")
            end
         
        end
    end
    
end

