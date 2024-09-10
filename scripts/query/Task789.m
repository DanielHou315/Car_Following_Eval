% Prepare the data for Task 789 Traffic congestion

RawTask789 = readtable('SQLprogram/Task789_AllBrakeEvents.csv','TreatAsEmpty',{'NULL'});
RawTask789{1,end+1}=0;
RawTask789.Properties.VariableNames{19} = 'SorE';

for i=1:size(RawTask789,1)-1
    if RawTask789.PriorMaxSpeed(i)>11.176 && ...%RawTask789.PriorMaxSpeed(i)>11.176 && ...
            RawTask789.MeanSpeed(i)<11.176 && ...%RawTask789.MaxSpeed(i)<11.176 && ...
            RawTask789.NextMaxSpeed(i)<11.176 && ...RawTask789.NextMaxSpeed(i)<11.176
            RawTask789.BrakeStart(i+1)-RawTask789.BrakeStart(i)<=9000
        RawTask789.SorE(i)=1;   % jam start
    elseif RawTask789.PriorMaxSpeed(i)<11.176 && ...%RawTask789.PriorMaxSpeed(i)<11.176 && ...
            RawTask789.MeanSpeed(i)<11.176 && ...%RawTask789.MaxSpeed(i)<11.176 && ...
            RawTask789.NextMaxSpeed(i)<11.176 && ...RawTask789.NextMaxSpeed(i)<11.176
            RawTask789.BrakeStart(i+1)-RawTask789.BrakeStart(i)<=9000
        RawTask789.SorE(i)=2;   % in the jam
    elseif RawTask789.PriorMaxSpeed(i)<11.176 && ...%RawTask789.PriorMaxSpeed(i)<11.176 && ...
            RawTask789.MeanSpeed(i)<11.176 && ...%RawTask789.MaxSpeed(i)<11.176 && ...
            RawTask789.NextMaxSpeed(i)>11.176 %&& RawTask789.NextMaxSpeed(i)>11.176
        RawTask789.SorE(i)=3;   % jame end
    else
        RawTask789.SorE(i)=0;
    end
end

KeepDT=[];
for j=1:115     % max driver id
    for k=1:477 % max trip id
        SorEGroup = RawTask789{RawTask789.Driver==j & RawTask789.Trip==k, 19};
        if ~isempty(SorEGroup)
            if all(ismember(1,SorEGroup)) && all(ismember(2,SorEGroup))&& all(ismember(3,SorEGroup))
                KeepDT=[KeepDT; j k];
            end
        end
    end
end

CandiTrip=RawTask789(ismember([RawTask789.Driver RawTask789.Trip],KeepDT,'rows'),:);

% select 1222~3 only
CandiTrip{1,end+1}=0;
CandiTrip.Properties.VariableNames{20} = 'JamNumber';

% numbering jam event
Sctrl=0;    % start found
Mctrl=0;    % middle found
Ectrl=0;
N=1;
for i=1:size(CandiTrip,1)-1
    if CandiTrip.SorE(i)==1 && CandiTrip.SorE(i+1)==2 && CandiTrip.BrakeEnd(i)-CandiTrip.BrakeStart(i)>=10   % brake start found
        Sctrl=1;
        Ectrl=0;
        SDriver=CandiTrip.Driver(i);
        STrip=CandiTrip.Trip(i);
        STime=CandiTrip.BrakeStart(i);  % jam start time
        SRow=i; % jam start row
    elseif CandiTrip.SorE(i)==2 && (CandiTrip.SorE(i-1)==1 || CandiTrip.SorE(i-1)==2) && ...
            (CandiTrip.SorE(i+1)==2 || CandiTrip.SorE(i+1)==3) && Sctrl==1 && CandiTrip.BrakeEnd(i)-CandiTrip.BrakeStart(i)>=10
        if CandiTrip.Driver(i)==SDriver && CandiTrip.Trip(i)==STrip
            Mctrl=1;
        else    % not same trip => reset numbers
            Sctrl=0;
            Mctrl=0;
            Ectrl=0;
        end
    elseif CandiTrip.SorE(i)==3 && CandiTrip.SorE(i-1)==2 && Sctrl==1 && Mctrl==1
        if CandiTrip.BrakeStart(i)-STime>=9000 && CandiTrip.Driver(i)==SDriver && CandiTrip.Trip(i)==STrip && ...
                CandiTrip.BrakeEnd(i)-CandiTrip.BrakeStart(i)>=10     % jam duration >= 1.5 min
            ERow=i; % jam end row
            Ectrl=1;
        else    % not long enough or not same trip => reset numbers and look for the start of next event
            Sctrl=0;
            Mctrl=0;
            Ectrl=0;
        end
    else
        Sctrl=0;
        Mctrl=0;
        Ectrl=0;
    end
    % fill in jam number
    if Sctrl==1 && Mctrl==1 && Ectrl==1
        CandiTrip{SRow:ERow,20}=N;
        N=N+1;
        Sctrl=0;
        Mctrl=0;
        Ectrl=0;
    end
end

QueryData = [CandiTrip{CandiTrip{:,20}~=0 & CandiTrip{:,19}==1, [1 2 5 9]} CandiTrip{CandiTrip{:,20}~=0 & CandiTrip{:,19}==3, [6 9 20]}];
QueryData = QueryData(:,[1 2 3 5 7 4 6]);

save('SQLprogram/Task789LabeledBrakeEvent.mat','CandiTrip');
save('SQLprogram/Task789ValidBrakeEvent.mat','QueryData');
