from datetime import datetime, timedelta
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import netCDF4 as nc
import matplotlib.pyplot as plt
class WaveHeightPredictor(nn.Module):
    def __init__(self):
        super(WaveHeightPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(21, 64),
             nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            
        )
        
       
    def forward(self, x):
        x=self.model(x)
        return x
if __name__=='__main__':
    
    
    num_samples=18000
    np.random.seed(365363)
    path1="/home/remecca/core/Ocean/EU_north_atlantic.nc"
    
    ocean_data=nc.Dataset(path1)

    lat_var = ocean_data.variables['lat']
    lon_var =ocean_data.variables['lon']
    time_var =ocean_data.variables['time']
    
    lats=np.random.uniform(-30,0,num_samples)
    lons=np.random.uniform(-175,-155,num_samples)
    #hours that will be added to the date 1/1/2018
    hours=np.random.uniform(19,8757-72,num_samples)
    #amount of time in the future waves will be predicted
    time_amount=np.random.uniform(1,73,num_samples)
    wave_heights_6=[]
    wave_heights_12=[]
    wave_heights_18=[]
    wave_heights=[]
    wave_periods_6=[]
    wave_periods_12=[]
    wave_periods_18=[]
    wave_periods=[]
    wave_directions_6=[]
    wave_directions_12=[]
    wave_directions_18=[]
    wave_directions=[]
    wind_directions_6=[]
    wind_directions_12=[]
    wind_directions_18=[]
    wind_directions=[]
    wind_speeds_6=[]
    wind_speeds_12=[]
    wind_speeds_18=[]
    wind_speeds=[]
    wave_height_correct=[]
    wave_heading_correct=[]
    wave_period_correct=[]
    for i in range(0,num_samples):
        time=datetime(2018,1,1)+timedelta(hours=hours[i])
        time_index = np.searchsorted(time_var[:], nc.date2num(time, units=time_var.units))
        time_index_6 = np.searchsorted(time_var[:], nc.date2num(time-timedelta(hours=6), units=time_var.units))
        time_index_12 = np.searchsorted(time_var[:], nc.date2num(time-timedelta(hours=12), units=time_var.units))
        time_index_18 = np.searchsorted(time_var[:], nc.date2num(time-timedelta(hours=18), units=time_var.units))
        lat_idx = np.abs(lat_var[:] - lats[i]).argmin()
        lon_idx = np.abs(lon_var[:] - lons[i]).argmin()
        wave_heights.append(ocean_data.variables['sig_wave_height'][time_index, lat_idx, lon_idx])
        wave_heights_6.append(ocean_data.variables['sig_wave_height'][time_index_6, lat_idx, lon_idx])
        wave_heights_12.append(ocean_data.variables['sig_wave_height'][time_index_12, lat_idx, lon_idx])
        wave_heights_18.append(ocean_data.variables['sig_wave_height'][time_index_18, lat_idx, lon_idx])
        wave_periods.append(ocean_data.variables['wave_period'][time_index, lat_idx, lon_idx])
        wave_periods_6.append(ocean_data.variables['wave_period'][time_index_6, lat_idx, lon_idx])
        wave_periods_12.append(ocean_data.variables['wave_period'][time_index_12, lat_idx, lon_idx])
        wave_periods_18.append(ocean_data.variables['wave_period'][time_index_18, lat_idx, lon_idx])
        wave_directions.append(ocean_data.variables['wave_direction'][time_index, lat_idx, lon_idx])
        wave_directions_6.append(ocean_data.variables['wave_direction'][time_index_6, lat_idx, lon_idx])
        wave_directions_12.append(ocean_data.variables['wave_direction'][time_index_12, lat_idx, lon_idx])
        wave_directions_18.append(ocean_data.variables['wave_direction'][time_index_18, lat_idx, lon_idx])
        wind_directions.append(ocean_data.variables['wind_direction'][time_index, lat_idx, lon_idx])
        wind_directions_6.append(ocean_data.variables['wind_direction'][time_index_6, lat_idx, lon_idx])
        wind_directions_12.append(ocean_data.variables['wind_direction'][time_index_12, lat_idx, lon_idx])
        wind_directions_18.append(ocean_data.variables['wind_direction'][time_index_18, lat_idx, lon_idx])
        wind_speeds_6.append(ocean_data.variables['wind_speed'][time_index_6, lat_idx, lon_idx])
        wind_speeds_12.append(ocean_data.variables['wind_speed'][time_index_12, lat_idx, lon_idx])
        wind_speeds_18.append(ocean_data.variables['wind_speed'][time_index_18, lat_idx, lon_idx])
        wind_speeds.append(ocean_data.variables['wind_speed'][time_index, lat_idx, lon_idx])    
        time_index = np.searchsorted(time_var[:], nc.date2num(time+timedelta(hours=time_amount[i]), units=time_var.units))
        wave_height_correct.append(ocean_data.variables['sig_wave_height'][time_index, lat_idx, lon_idx])
        wave_heading_correct.append(ocean_data.variables['wave_direction'][time_index, lat_idx, lon_idx])
        wave_period_correct.append(ocean_data.variables['wave_period'][time_index, lat_idx, lon_idx])
        time_index_48 = np.searchsorted(time_var[:], nc.date2num(time+timedelta(hours=48), units=time_var.units))
    wave_height_correct = np.array([float(x) for x in wave_height_correct])
    wave_heading_correct = np.array([float(x) for x in wave_heading_correct])
    wave_period_correct = np.array([float(x) for x in wave_period_correct])
    wave_heading_radians = np.radians(wave_heading_correct)
    wave_heading_sin = np.sin(wave_heading_radians)
    wave_heading_cos = np.cos(wave_heading_radians)
    
    training_data_input=torch.tensor(np.column_stack((wave_heights[0:(int)(num_samples*0.7)], wave_heights_18[0:(int)(num_samples*0.7)],
    wave_heights_12[0:(int)(num_samples*0.7)],wave_heights_6[0:(int)(num_samples*0.7)],
    wave_periods[0:(int)(num_samples*0.7)], wave_periods_18[0:(int)(num_samples*0.7)],
    wave_periods_12[0:(int)(num_samples*0.7)],wave_periods_6[0:(int)(num_samples*0.7)],
    wave_directions[0:(int)(num_samples*0.7)], wave_directions_18[0:(int)(num_samples*0.7)],
    wave_directions_12[0:(int)(num_samples*0.7)],wave_directions_6[0:(int)(num_samples*0.7)],
    wind_directions[0:(int)(num_samples*0.7)], wind_directions_18[0:(int)(num_samples*0.7)],
    wind_directions_12[0:(int)(num_samples*0.7)],wind_directions_6[0:(int)(num_samples*0.7)],
    wind_speeds[0:(int)(num_samples*0.7)],wind_speeds_18[0:(int)(num_samples*0.7)],
    wind_speeds_12[0:(int)(num_samples*0.7)],wind_speeds_6[0:(int)(num_samples*0.7)],time_amount[0:(int)(num_samples*0.7)])), dtype=torch.float32)

    training_data_correct_out_real=torch.tensor(np.column_stack((wave_height_correct[0:(int)(num_samples*0.7)],
    wave_heading_sin[0:(int)(num_samples*0.7)],wave_heading_cos[0:(int)(num_samples*0.7)],wave_period_correct[0:(int)(num_samples*0.7)])), dtype=torch.float32)
    #training_data_correct_out_real=torch.tensor((wave_height_correct[0:(int)(num_samples*0.7)]), dtype=torch.float32).view(-1,1) 
    validation_data_input=torch.tensor(np.column_stack((wave_heights[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_heights_18[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wave_heights_12[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_heights_6[(int)(num_samples*0.7):(int)(num_samples*0.9)], 
    wave_periods[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_periods_18[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wave_periods_12[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_periods_6[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wave_directions[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_directions_18[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wave_directions_12[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_directions_6[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wind_directions[(int)(num_samples*0.7):(int)(num_samples*0.9)],wind_directions_18[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wind_directions_12[(int)(num_samples*0.7):(int)(num_samples*0.9)],wind_directions_6[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wind_speeds[(int)(num_samples*0.7):(int)(num_samples*0.9)],wind_speeds_18[(int)(num_samples*0.7):(int)(num_samples*0.9)],
    wind_speeds_12[(int)(num_samples*0.7):(int)(num_samples*0.9)],wind_speeds_6[(int)(num_samples*0.7):(int)(num_samples*0.9)],time_amount[(int)(num_samples*0.7):(int)(num_samples*0.9)])), dtype=torch.float32) 
    
    validation_data_correct_out_real=torch.tensor(np.column_stack((wave_height_correct[(int)(num_samples*0.7):(int)(num_samples*0.9)]
    ,wave_heading_sin[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_heading_cos[(int)(num_samples*0.7):(int)(num_samples*0.9)],wave_period_correct[(int)(num_samples*0.7):(int)(num_samples*0.9)])), dtype=torch.float32)
    #validation_data_correct_out_real=torch.tensor((wave_height_correct[(int)(num_samples*0.7):(int)(num_samples*0.9)]), dtype=torch.float32).view(-1,1)
    
    

    # Convert data to PyTorch tensors 
    training_data_input = training_data_input.float()
    training_data_correct_out = training_data_correct_out_real.float()

    
    torch.save(training_data_correct_out_real, "training_data_correct_out_real.pt")
    torch.save(training_data_input, "training_data_input.pt")
    torch.save(validation_data_correct_out_real, "validation_data_correct_out_real.pt")
    torch.save(validation_data_input, "validation_data_input.pt")  
    
    
    
    training_data_correct_out_real=torch.load("training_data_correct_out_real.pt")
    training_data_input=torch.load("training_data_input.pt")
    validation_data_correct_out_real=torch.load("validation_data_correct_out_real.pt")
    validation_data_input=torch.load("validation_data_input.pt")
    train_mean = training_data_input.mean(dim=0, keepdim=True)
    train_std = training_data_input.std(dim=0, keepdim=True)
    normalization = {
    'train_mean': train_mean,
    'train_std': train_std,
}
    torch.save(normalization, 'normalization.pt')
    # Normalize training data
    training_data_input = (training_data_input - train_mean) / train_std

  # Apply the same normalization to validation and test data
    validation_data_input = (validation_data_input - train_mean) / train_std
    
    
    model = WaveHeightPredictor()
    
    num_epochs = 800 # Number of training iterations
    losses=[]
    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='none')  # Mean Squared Error for regression
    
    def weighted_mse_loss(pred, target):
    # Separate components
     pred_height = pred[:, 0]
     pred_heading_sin =pred[:, 1]
     pred_heading_cos =pred[:, 2]
     pred_period = pred[:, 3]

     target_height = target[:, 0]
     target_heading_sin = target[:, 1]
     target_heading_cos = target[:, 2]
     target_period = target[:, 3]
   
     pred_angle = torch.atan2(pred_heading_sin, pred_heading_cos)  # [-pi, pi]
     
     true_angle = torch.atan2(target_heading_sin, target_heading_cos)
     target_angle_deg = (torch.rad2deg(true_angle) + 360) % 360

     heading_weight = torch.where(
     (target_angle_deg >= 60) & (target_angle_deg <= 200),
     torch.tensor(20.0, device=target.device),  # upweight this range
    torch.tensor(1.0, device=target.device)   # normal weight elsewhere
    )

     angle_diff = (pred_angle - true_angle + np.pi) % (2 * np.pi) - np.pi
     heading_loss = torch.mean(criterion(pred_angle, true_angle))
    
     # Weighted height loss
     weights = torch.exp(target_height / 4.0)
     height_loss = criterion(pred_height, target_height)
     weighted_height_loss = torch.mean(weights * height_loss)

     # Regular MSE for period
     period_loss = torch.mean(criterion(pred_period, target_period))

     # Weighted sum of losses
     total_loss = (
        1.0 * weighted_height_loss +
        1.0 * period_loss +
        1.0 * heading_loss  # you can adjust this factor
    )

     return total_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

    for epoch in range(num_epochs):
     # Forward pass
     predictions = model(training_data_input)
     loss = weighted_mse_loss(predictions, training_data_correct_out_real)  # Compute loss (mean squared error)
     
     # Backward pass
     optimizer.zero_grad()  # Reset gradients
     loss.backward()  # Compute gradients
     optimizer.step()  # Update weights
     losses.append(loss.item())
     # Print loss every 10 epochs
     if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Final Training Loss: {loss.item():.4f}')
    
    # Switch model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()

    # Predict on validation data
    with torch.no_grad():
     print(validation_data_input.shape)
     val_predictions = model(validation_data_input)
     val_loss = weighted_mse_loss(val_predictions, validation_data_correct_out_real)
    
    print(f'Validation Loss: {val_loss.item():.4f}')
    torch.save(model.state_dict(), 'model.pth')
    valdata=validation_data_correct_out_real.numpy()
    training_data=training_data_correct_out_real.numpy()
    val_time_amounts=validation_data_input[:, -1].numpy()
    # Reverse normalization
    val_time_amounts = val_time_amounts * train_std[0,-1].item() + train_mean[0,-1].item()
    height_training_plot=[]
    heading_training_plot=[]
    valpred=val_predictions.numpy()
    val_dataplot_height=[]
    val_predplot_height=[]
    val_dataplot_heading=[]
    val_predplot_heading=[]
    val_dataplot_period=[]
    val_predplot_period=[]
    val_diff=[]    
    for i in range(0, len(training_data)):
       height_training_plot.append(training_data[i][0])
       cos_data=training_data[i][2]
       sin_data=training_data[i][1]
       data_rad = np.arctan2(sin_data, cos_data)
       heading_training_plot.append(np.degrees(data_rad) % 360)
       
    
    for i in range(0,len(val_predictions)):
       val_dataplot_height.append(valdata[i][0])
       val_predplot_height.append(valpred[i][0])
       sin_data=valdata[i][1]
       sin_pred=valpred[i][1]
       cos_data=valdata[i][2]
       cos_pred=valpred[i][2]
       data_rad = np.arctan2(sin_data, cos_data)
       pred_rad = np.arctan2(sin_pred, cos_pred)
       val_dataplot_heading.append(np.degrees(data_rad) % 360)
       val_predplot_heading.append(np.degrees(pred_rad) % 360)
       val_dataplot_period.append(valdata[i][3])
       val_predplot_period.append(valpred[i][3])

       '''
       #print("Real Wave Height")
       print(valdata[i][0])
       print("Predicted Wave Height")
       print(valpred[i][0])
       diff=valpred[i][0]-valdata[i][0]
       val_diff.append(diff)
       print("Difference")
       print(diff)
       print("    ")
       '''
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(val_dataplot_height, val_predplot_height, c=val_time_amounts, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label="Time Interval (hours)")
    plt.plot([min(val_dataplot_height), max(val_dataplot_height)],
         [min(val_dataplot_height), max(val_dataplot_height)], 'r--', label="Ideal Fit (y=x)")
    plt.title("Predicted Vs. Actual Wave Height")
    plt.xlabel("Actual Wave Height (meters)")
    plt.ylabel("Predicted Wave Height (meters)")
    plt.legend()
    plt.show()

# --- Histogram: Wave Height ---
    plt.figure(figsize=(8, 6))
    bins = np.linspace(min(min(val_dataplot_height), min(val_predplot_height)),
                   max(max(val_dataplot_height), max(val_predplot_height)), 31)
    plt.hist(val_dataplot_height, bins=bins, alpha=0.5, label='Actual', edgecolor='black')
    plt.hist(val_predplot_height, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
    plt.title('Predicted vs. Actual Wave Heights')
    plt.xlabel('Wave Height (meters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



    actual_radians = np.deg2rad(val_dataplot_heading)
    predicted_radians = np.deg2rad(val_predplot_heading)
    errors = np.abs(np.rad2deg(np.arctan2(np.sin(predicted_radians - actual_radians),
                                      np.cos(predicted_radians - actual_radians))))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    sc = ax.scatter(actual_radians, errors, c=errors, cmap='coolwarm', alpha=0.8)
    plt.colorbar(sc, label='Prediction Error (degrees)')
    ax.set_title("Error Magnitude at Actual Heading")
    plt.show()

# --- Histogram: Wave Heading ---
    plt.figure(figsize=(8, 6))
    bins = np.linspace(min(min(val_dataplot_heading), min(val_predplot_heading)),
                   max(max(val_dataplot_heading), max(val_predplot_heading)), 31)
    plt.hist(val_dataplot_heading, bins=bins, alpha=0.5, label='Actual', edgecolor='black')
    plt.hist(val_predplot_heading, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
    plt.title('Predicted vs. Actual Wave Heading')
    plt.xlabel('Wave Heading (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# --- Scatter: Wave Period ---
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(val_dataplot_period, val_predplot_period, c=val_time_amounts, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label="Time Interval (hours)")
    plt.plot([min(val_dataplot_period), max(val_dataplot_period)],
         [min(val_dataplot_period), max(val_dataplot_period)], 'r--', label="Ideal Fit (y=x)")
    plt.title("Predicted Vs. Actual Wave Period")
    plt.xlabel("Actual Wave Period")
    plt.ylabel("Predicted Wave Period")
    plt.legend()
    plt.show()

# --- Histogram: Wave Period ---
    plt.figure(figsize=(8, 6))
    print("val_dataplot_period min/max:", min(val_dataplot_period), max(val_dataplot_period))
    print("val_predplot_period min/max:", min(val_predplot_period), max(val_predplot_period))
    bins = np.linspace(min(min(val_dataplot_period), min(val_predplot_period)),
                   max(max(val_dataplot_period), max(val_predplot_period)), 31)
    print("bins:", bins)
    plt.hist(val_dataplot_period, bins=bins, alpha=0.5, label='Actual', edgecolor='black')
    plt.hist(val_predplot_period, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
    plt.title('Predicted vs. Actual Wave Periods')
    plt.xlabel('Wave Period')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
 
 #Training data Histogram
    '''
    ax8.hist(height_training_plot, bins=30, alpha=0.5, edgecolor='black')
    ax8.set_title('Training Wave Heights')
    ax8.set_xlabel('Wave Height')
    ax8.set_ylabel('Frequency')

    ax4.hist(heading_training_plot, bins=30, alpha=0.5, edgecolor='black')
    ax4.set_title('Training Headings')
    ax4.set_xlabel('Wave Heading')
    ax4.set_ylabel('Frequency')
    '''
    plt.tight_layout()
    plt.show()