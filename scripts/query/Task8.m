% Deceleration profiles

%% data table for Task 8
% how to decelerate
BrakeEvent = CandiTrip{CandiTrip{:,20}~=0,[1 2 5 6]};   % driver, trip, brakestart, brakeend
Task8Data=[];
for B=1:size(BrakeEvent,1)
    % 1=speed at brakestart; 2=delta speed between brakestart and brakeend;
    % 3=mean Ax between brakestart and brakeend;
    % 4=delta Ax between brakestart and brakestart+10
    % 5=delta Ax between brakeend-10 and brakeend
    % 6=brake reason(1=lead braking, 2=target changed)
    VAtBrake = Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,3),4);
    DV = Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,4),4)-...
        Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,3),4);
    MeanAx = mean(Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)>=BrakeEvent(B,3) & Jam(:,3)<=BrakeEvent(B,4),6));
    JerkAtBS = (Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,3)+10,6)-...
            Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,3),6))*10;
    JerkAtBE = (Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,4),6)-...
            Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)==BrakeEvent(B,4)-10,6))*10;
    TargetSeq = Jam(Jam(:,1)==BrakeEvent(B,1) & Jam(:,2)==BrakeEvent(B,2) & Jam(:,3)>=BrakeEvent(B,3) & Jam(:,3)<=BrakeEvent(B,4),10);
    if length(TargetSeq(TargetSeq(:,1)==TargetSeq(1,1),1))==length(TargetSeq)
        BrakeReason=1;
    else
        BrakeReason=2;
    end
        
    Task8Data=[Task8Data; VAtBrake, DV, MeanAx, JerkAtBS, JerkAtBE, BrakeReason];
    
end
Task8Data(:,[1 2]) = Task8Data(:,[1 2])*2.23694;

%% at deceleration
% Task8; VAtBrake, DV, MeanAx, JerkAtBS, JerkAtBE, BrakeReason
% speed at the start of brake
figure
histogram(Task8Data(:,1)*2.23694)
xlabel('Speed at the start of brake (MPH)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task8Data(:,2)*2.23694)
xlabel('Speed difference between the start/end of brake (MPH)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task8Data(:,3))
xlabel('Mean deceleration within a brake event (m/s2)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task8Data(:,4))
xlabel('Jerk at the start of brake (m/s3)');
ylabel('Frequency');
v=gca;
v.FontSize=16;

figure
histogram(Task8Data(:,5))
xlabel('Jerk at the end of brake (m/s3)');
ylabel('Frequency');
v=gca;
v.FontSize=16;
%%
figure
histogram(Task8Data(:,6))
xlabel('Reason of deceleration');
xticks([1 2]);
xticklabels({'Lead vehicle braking','Other vehicle merging'});
ylabel('Frequency');
v=gca;
v.FontSize=16;


%% heatmap: DV vs VAtBrake
% Task8Data: VAtBrake, DV, MeanAx, JerkAtBS, JerkAtBE, BrakeReason
[X,Y]=meshgrid(-40:1:0, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task8Data(:,2)>=X(i,j) & Task8Data(:,2)<X(i,j)+1 & Task8Data(:,1)>=Y(i,j) & Task8Data(:,1)<Y(i,j)+1));
    end
end

image([-40 0],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Speed difference (MPH)');
ylabel('Speed at the start of brake (MPH)');

%% heatmap: MeanAx vs VAtBrake
[X,Y]=meshgrid(-2:0.1:2, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task8Data(:,3)>=X(i,j) & Task8Data(:,3)<X(i,j)+0.1 & Task8Data(:,1)>=Y(i,j) & Task8Data(:,1)<Y(i,j)+1));
    end
end

image([-2 2],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Mean deceleration within a brake event (m/s2)');
ylabel('Speed at the start of brake (MPH)');

%% heatmap: JerkStart vs VAtBrake
[X,Y]=meshgrid(-3:0.1:3, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task8Data(:,3)>=X(i,j) & Task8Data(:,3)<X(i,j)+0.1 & Task8Data(:,1)>=Y(i,j) & Task8Data(:,1)<Y(i,j)+1));
    end
end

image([-3 3],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Jerk at the start of brake (m/s3)');
ylabel('Speed at the start of brake (MPH)');

%% heatmap: JerkEnd vs VAtBrake
[X,Y]=meshgrid(-5:0.2:5, 0:1:50);
M=zeros(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        M(i,j)= length(find(Task8Data(:,3)>=X(i,j) & Task8Data(:,3)<X(i,j)+0.2 & Task8Data(:,1)>=Y(i,j) & Task8Data(:,1)<Y(i,j)+1));
    end
end

image([-5 5],[0 50],M,'CDataMapping','scaled');
set(gca,'YDir','normal')
myColorMap = jet(256);
myColorMap(1,:)=1;
colormap(myColorMap);
colorbar
xlabel('Jerk at the end of brake (m/s3)');
ylabel('Speed at the start of brake (MPH)');
xticks([-5:1:5]);

%% save csv files
Task8Table = array2table(Task8Data,'VariableNames',{'SpeedAtBrake','SpeedDifference','MeanDeceleration','JerkAtBrakeStart','JerkAtBrakeEnd','BrakeReason'});
