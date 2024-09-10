% Acceleration profiles

% select acceleration events between brake events
JamS_E = CandiTrip(CandiTrip{:,20}~=0,:);
NoBrakeEvent = [JamS_E{1:end-1,[1 2 6 9]} JamS_E{2:end,5}];
NoBrakeEvent = NoBrakeEvent(:,[1 2 3 5 4]);   % reorder

% create acceleration events
NonBrakeS_E = [];
for i=1:size(NoBrakeEvent,1)
    NonBrakeS_E = [NonBrakeS_E; Jam(Jam(:,1)==NoBrakeEvent(i,1) & Jam(:,2)==NoBrakeEvent(i,2)...
        & Jam(:,3)>=NoBrakeEvent(i,3) & Jam(:,3)<=NoBrakeEvent(i,4),:)];
end
AccelData = NonBrakeS_E(NonBrakeS_E(:,5)~=0,:);
N=1;
AccelData(1,end+1)=N;  % event number
AccelData(1,end+1)=1;  % event start flag
AccelData(end,end)=3;  % event end flag
for i=2:size(AccelData,1)-1    % from 2nd to last 2nd
    if AccelData(i,3)-AccelData(i-1,3)==10    % consecutive time units
        AccelData(i,end-1)=N;        
    elseif AccelData(i,3)-AccelData(i-1,3)~=10 && AccelData(i+1,3)-AccelData(i,3)==10
        N=N+1;
        AccelData(i,end-1)=N;
    end
    % set start/end flags
    if AccelData(i,end-1)==AccelData(i-1,end-1) && AccelData(i,3)-AccelData(i+1,3)==-10
        AccelData(i,end)=2;    % same event
    elseif AccelData(i,end-1)~=AccelData(i-1,end-1) && AccelData(i,15)~=0
        AccelData(i,end)=1;    % new event
    elseif AccelData(i,end-1)==AccelData(i-1,end-1) && AccelData(i,15)~=0
        AccelData(i,end)=3;    % event end
    end
end
AccelData(end,end-1)=N;  % last event number

%%
Task9Data=[];
for C=1:AccelData(end,end-1)    % to the last accel event number
    % 1=speed at throttle applied; 2=delta speed between throttle applied and throttle released;
    % 3=mean Ax between brakestart and brakeend;
    % 4=delta Ax between brakestart and brakestart+10
    % 5=delta Ax between brakeend-10 and brakeend
    % 6=brake reason(1=lead braking, 2=target changed)
    VAtAccel = AccelData(AccelData(:,15)==C & AccelData(:,16)==1,4);
    DV_acc = AccelData(AccelData(:,15)==C & AccelData(:,16)==3,4) - AccelData(AccelData(:,15)==C & AccelData(:,16)==1,4);
    MeanAx_acc = mean(AccelData(find(AccelData(:,15)==C & AccelData(:,16)==1):find(AccelData(:,15)==C & AccelData(:,16)==3),6));
    JerkAtAS = (AccelData(find(AccelData(:,15)==C & AccelData(:,16)==1)+1,6) - AccelData(AccelData(:,15)==C & AccelData(:,16)==1,6))*10;
    JerkAtAE = (AccelData(AccelData(:,15)==C & AccelData(:,16)==3,6) - AccelData(find(AccelData(:,15)==C & AccelData(:,16)==3)-1,6))*10;
    TargetSeq_acc = AccelData(find(AccelData(:,15)==C & AccelData(:,16)==1):find(AccelData(:,15)==C & AccelData(:,16)==3),10);
    if length(TargetSeq_acc(TargetSeq_acc(:,1)==TargetSeq_acc(1,1),1))==length(TargetSeq_acc)
        AccelReason=1;
    else
        AccelReason=2;
    end
    Task9Data=[Task9Data; VAtAccel, DV_acc, MeanAx_acc, JerkAtAS, JerkAtAE, AccelReason];
    
end
Task9Data(:,[1 2]) = Task9Data(:,[1 2])*2.23694;

%% at acceleration
% Task9; VAtAccel, DV_acc, MeanAx_acc, JerkAtAS, JerkAtAE, AccelReason
% speed at the start of brake
figure
histogram(Task9Data(:,1)*2.23694)
xlabel('Speed at the start of acceleration (MPH)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task9Data(:,2)*2.23694)
xlabel({'Speed difference between the start/end of';'acceleration (MPH)'});
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task9Data(:,3))
xlabel({'Mean acceleration within';'an acceleration event (m/s2)'});
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task9Data(:,4))
xlabel('Jerk at the start of acceleration (m/s3)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task9Data(:,5))
xlabel('Jerk at the end of acceleration (m/s3)');
ylabel('Frequency');
v=gca;
v.FontSize=16;
%%
figure
histogram(Task9Data(:,6))
xlabel('Reason of acceleration');
xticks([1 2]);
xticklabels({'Lead vehicle accelerating','Lead vehicle cutting out'});
ylabel('Frequency');
v=gca;
v.FontSize=16;


%% heatmap: DV vs VAtBrake
[X,Y]=meshgrid(0:1:40, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task9Data(:,2)>=X(i,j) & Task9Data(:,2)<X(i,j)+1 & Task9Data(:,1)>=Y(i,j) & Task9Data(:,1)<Y(i,j)+1));
    end
end

image([0 40],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Speed difference (MPH)');
ylabel('Speed at the start of acceleration (MPH)');

%% heatmap: MeanAx vs VAtBrake
[X,Y]=meshgrid(-1:0.1:3, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task9Data(:,3)>=X(i,j) & Task9Data(:,3)<X(i,j)+0.1 & Task9Data(:,1)>=Y(i,j) & Task9Data(:,1)<Y(i,j)+1));
    end
end

image([-1 3],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Mean acceleration within a brake event (m/s2)');
ylabel('Speed at the start of acceleration (MPH)');

%% heatmap: JerkStart vs VAtBrake
[X,Y]=meshgrid(-3:0.1:3, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task9Data(:,3)>=X(i,j) & Task9Data(:,3)<X(i,j)+0.1 & Task9Data(:,1)>=Y(i,j) & Task9Data(:,1)<Y(i,j)+1));
    end
end

image([-3 3],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Jerk at the start of acceleration (m/s3)');
ylabel('Speed at the start of acceleration (MPH)');

%% heatmap: JerkEnd vs VAtBrake
[X,Y]=meshgrid(-5:0.2:5, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task9Data(:,3)>=X(i,j) & Task9Data(:,3)<X(i,j)+0.2 & Task9Data(:,1)>=Y(i,j) & Task9Data(:,1)<Y(i,j)+1));
    end
end

image([-5 5],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Jerk at the end of acceleration (m/s3)');
ylabel('Speed at the start of acceleration (MPH)');
xticks([-5:1:5]);

%% save csv files
Task9Table = array2table(Task9Data,'VariableNames',{'SpeedAtBrake','SpeedDifference','MeanAcceleration','JerkAtAccelerationStart','JerkAtAccelerationEnd','AccelerationReason'});