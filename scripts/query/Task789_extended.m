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

%% query data between 1 and 3
Jam=[];
% Specify the name of the user data source 
DataSourceName = 'ESG';

% Connect to the database
conn = database(DataSourceName,'','');

% Set database preferences
setdbprefs('DataReturnFormat','numeric');

for Q=1:size(QueryData,1)
    disp(Q)
    Jam_Q = strcat('select A.*, B.Ax, C.Range, C.RangeRate, C.TTC, C.TargetId, D.TotalCount TrafficCount, E.AgeGroup, E.Gender from',...
    ' (select Driver, Trip, Time, Speed, Distance, AccelPedal from LvFot..Data where Driver=',num2str(QueryData(Q,1))," ",'and Trip=',num2str(QueryData(Q,2)),...
    ' and Time between'," ",num2str(QueryData(Q,3))," ",'and'," ",num2str(QueryData(Q,4)),') A',...
    ' join',...
    ' (select Driver, Trip, Time, Ax from LvFot..Imu) B on A.Driver=B.Driver and A.Trip=B.Trip and A.Time=B.Time',...
    ' join',...
    ' (select Driver, Trip, Time, TargetId, Range, RangeRate, TTC from LvPub..CipvThConflict) C',...
    ' on A.Driver=C.Driver and A.Trip=C.Trip and A.Time=C.Time',...
    ' join',...
    ' (select * from LvPub..TrafficCount) D on A.Driver=D.Driver and A.Trip=D.Trip and A.Time=D.Time',...
    ' join',...
    ' (select * from LvFot..ValidTrips) E on A.Driver=E.Driver and A.Trip=E.Trip');

% Execute the SQL statement, and returns a cursor object
curs = exec(conn, Jam_Q);

% Import data into MATLAB
curs = fetch(curs);

% Save the data to a MATLAB array
temp_Jam = curs.Data;
Jam = [Jam; temp_Jam];

clear('curs.Data');
close(curs);
end
% Jam header:
% 1=driver; 2=trip; 3=time; 4=speed; 5=distance, 6=accelpedal; 7=Ax; 8=range;
% 9=rangerate; 10=TTC; 11=targetid; 12=traffic count; 13=age; 14=gender;
save('SQLprogram/Task789Data.mat','Jam');



% Additional code to query and separate data into JamBefore and JamAfter

% Specify the name of the user data source 
DataSourceName = 'ESG';

% Connect to the database
conn = database(DataSourceName,'','');

% Set database preferences
setdbprefs('DataReturnFormat','numeric');

JamBefore = [];
JamAfter = [];

for Q = 1:size(QueryData,1)
    disp(Q)
    startTimeBefore = max(QueryData(Q,3) - 30000, 1);
    endTimeBefore = QueryData(Q,3) - 1;
    startTimeAfter = QueryData(Q,4) + 1;
    endTimeAfter = QueryData(Q,4) + 30000;

    % Query for the "before" data
    JamBefore_Q = strcat('select A.*, B.Ax, C.Range, C.RangeRate, C.TTC, C.TargetId, D.TotalCount TrafficCount, E.AgeGroup, E.Gender from', ...
    ' (select Driver, Trip, Time, Speed, Distance, AccelPedal from LvFot..Data where Driver=', num2str(QueryData(Q,1)), ' and Trip=', num2str(QueryData(Q,2)), ...
    ' and Time between ', num2str(startTimeBefore), ' and ', num2str(endTimeBefore), ') A', ...
    ' join', ...
    ' (select Driver, Trip, Time, Ax from LvFot..Imu) B on A.Driver=B.Driver and A.Trip=B.Trip and A.Time=B.Time', ...
    ' join', ...
    ' (select Driver, Trip, Time, TargetId, Range, RangeRate, TTC from LvPub..CipvThConflict) C on A.Driver=C.Driver and A.Trip=C.Trip and A.Time=C.Time', ...
    ' join', ...
    ' (select * from LvPub..TrafficCount) D on A.Driver=D.Driver and A.Trip=D.Trip and A.Time=D.Time', ...
    ' join', ...
    ' (select * from LvFot..ValidTrips) E on A.Driver=E.Driver and A.Trip=E.Trip');

    % Execute the SQL statement for "before" data and return a cursor object
    curs_before = exec(conn, JamBefore_Q);

    % Import data into MATLAB
    curs_before = fetch(curs_before);

    % Save the "before" data to a MATLAB array
    temp_JamBefore = curs_before.Data;
    JamBefore = [JamBefore; temp_JamBefore];

    clear('curs_before.Data');
    close(curs_before);

    % Query for the "after" data
    JamAfter_Q = strcat('select A.*, B.Ax, C.Range, C.RangeRate, C.TTC, C.TargetId, D.TotalCount TrafficCount, E.AgeGroup, E.Gender from', ...
    ' (select Driver, Trip, Time, Speed, Distance, AccelPedal from LvFot..Data where Driver=', num2str(QueryData(Q,1)), ' and Trip=', num2str(QueryData(Q,2)), ...
    ' and Time between ', num2str(startTimeAfter), ' and ', num2str(endTimeAfter), ') A', ...
    ' join', ...
    ' (select Driver, Trip, Time, Ax from LvFot..Imu) B on A.Driver=B.Driver and A.Trip=B.Trip and A.Time=B.Time', ...
    ' join', ...
    ' (select Driver, Trip, Time, TargetId, Range, RangeRate, TTC from LvPub..CipvThConflict) C on A.Driver=C.Driver and A.Trip=C.Trip and A.Time=C.Time', ...
    ' join', ...
    ' (select * from LvPub..TrafficCount) D on A.Driver=D.Driver and A.Trip=D.Trip and A.Time=D.Time', ...
    ' join', ...
    ' (select * from LvFot..ValidTrips) E on A.Driver=E.Driver and A.Trip=E.Trip');

    % Execute the SQL statement for "after" data and return a cursor object
    curs_after = exec(conn, JamAfter_Q);

    % Import data into MATLAB
    curs_after = fetch(curs_after);

    % Save the "after" data to a MATLAB array
    temp_JamAfter = curs_after.Data;
    JamAfter = [JamAfter; temp_JamAfter];

    clear('curs_after.Data');
    close(curs_after);
end

% JamBefore header:
% 1=driver; 2=trip; 3=time; 4=speed; 5=distance, 6=accelpedal; 7=Ax; 8=range;
% 9=rangerate; 10=TTC; 11=targetid; 12=traffic count; 13=age; 14=gender;

% JamAfter header:
% 1=driver; 2=trip; 3=time; 4=speed; 5=distance, 6=accelpedal; 7=Ax; 8=range;
% 9=rangerate; 10=TTC; 11=targetid; 12=traffic count; 13=age; 14=gender;

save('SQLprogram/Task789DataBefore.mat','JamBefore');
save('SQLprogram/Task789DataAfter.mat','JamAfter');













