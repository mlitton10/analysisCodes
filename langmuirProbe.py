import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

plt.style.use('/home/matt/latex_and_matplotlib_styles/matplotlib_styles/physrev.mplstyle')  # Set full path to
# physrev.mplstyle if the file is not in the same in directory as the notebook
plt.rcParams['figure.dpi'] = "300"

m_p = 938e6

def find_closest_point(array, item):
    return np.argmin(np.abs(array - item))


def r_squared(y_true, y_pred):
    # Calculate the residual sum of squares (SSres)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the total sum of squares (SStot)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Calculate R-squared
    r2 = 1 - (ss_res / ss_tot)
    return r2


class LangmuirProbe:
    def __init__(self, file_name, R, attenuation_factor, diameter, amu, r_squared_cut=0.98):
        self.R = R  # resistance to obtain true current of langmuir probe
        self.attenuation_factor = attenuation_factor  # scope attenuation and gain factor for sweep voltages
        self.probe_area = np.pi * (diameter / 2) ** 2  #  from probe construction, not accounting for sheath
        self.amu = amu  # mass of the fueling gas

        # get raw data from file
        raw_voltage, raw_current, raw_time, raw_positions = self.get_data(file_name)
        self.positions = raw_positions
        self.sweep_voltage, self.probe_current, self.probe_time, positions_dict = self.format_data(raw_voltage,
                                                                                                   raw_current,
                                                                                                   raw_time,
                                                                                                   raw_positions)

        self.n_x = positions_dict['n_x']
        self.n_y = positions_dict['n_y']
        self.x_positions = positions_dict['x_positions']
        self.y_positions = positions_dict['y_positions']

        self.te_contour = self.compute_plane_te(r_squared_cut)
        self.I_sat_plane = self.compute_I_sat_plane()
        self.n_e = self.compute_electron_density()
        pass

    def get_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            #  This is an issue, there is no gurantee which channel is which.
            #  Better solution would be to have some label we could pull. Ideally standardized. Not much I can do here
            #  Without updating the DAQ code. Maybe standard labels and formats for each probe type.
            raw_voltage = f['Acquisition']['LeCroy_scope']['Channel1'][()]
            raw_current = f['Acquisition']['LeCroy_scope']['Channel3'][()]
            raw_time = f['Acquisition']['LeCroy_scope']['time'][()]
            raw_positions = f['Control']['Positions']['positions_setup_array'][()]
        return raw_voltage, raw_current, raw_time, raw_positions

    def format_data(self, voltage_data, current_data, time, positions):
        n_t = len(time)
        #  Trim the data to cut it off before the sweep ends
        filtered_data = savgol_filter(current_data, int(0.004 * n_t), 5, axis=0)
        gradient = np.diff(filtered_data, axis=-1)
        sweep_off = np.argmin(gradient, axis=1) - int(0.005 * n_t)
        sweep_off_idx = int(np.mean(sweep_off))

        #  Scale and cut data appropriately
        sweep_voltage = voltage_data[:, :sweep_off_idx] * self.attenuation_factor
        probe_current = current_data[:, :sweep_off_idx] / self.R
        probe_time = time[:sweep_off_idx] * 1e6

        # I don't like how the positions are stored, and I need some code to better collect the position data
        all_x = []
        all_y = []

        for pos in positions:
            all_x.append(pos[1])
            all_y.append(pos[2])

        x_positions = np.unique(all_x)
        y_positions = np.unique(all_y)

        n_x = len(x_positions)
        n_y = len(y_positions)

        positions_dict = {'n_x': n_x, 'n_y': n_y, 'x_positions': x_positions, 'y_positions': y_positions}

        return sweep_voltage, probe_current, probe_time, positions_dict

    def compute_characteristic_te(self, voltage, current, r_squared_cut):

        idx_sort = np.argsort(voltage)
        voltage = voltage[idx_sort]
        current = current[idx_sort]

        non_zero_max = np.max(np.abs(np.diff(voltage))[np.where(np.abs(np.diff(voltage)) > 1e-8)])
        gradient = np.diff(gaussian_filter1d(current, int(0.008 * len(current))))

        idx_max = np.argmax(gradient)

        if non_zero_max > 0.01:
            delta_0 = non_zero_max
        else:
            delta_0 = 0.01  # volts

        delta_max = 50 * delta_0  # volts
        te_values = []
        r_squared_data = []

        for i in range(1, int(delta_max / delta_0)):
            delta = delta_0 * i  # define an interval around the peak gradient in the characteristic
            ind1, ind2 = find_closest_point(voltage, voltage[idx_max] - delta), find_closest_point(voltage,
                                                                                                   voltage[
                                                                                                       idx_max] + delta)

            # compute a linear fit in this domain
            m, b = np.polyfit(voltage[ind1:ind2], np.log(current[ind1:ind2]), 1)
            y_fit = m * voltage[ind1:ind2] + b

            # compute the quality of the linear fit
            r_squared_data.append(r_squared(np.log(current[ind1:ind2]), y_fit))

            te_values.append(1 / m)  # compute the temperature

        te_values = np.array(te_values)
        quality_fit = np.where(np.array(r_squared_data) > r_squared_cut)
        i = 1
        while len(quality_fit[0]) < 4:
            quality_fit = np.where(np.array(r_squared_data) > r_squared_cut - i * 0.01)
            i += 1
        if i > 10:
            te_final = np.nan
        else:
            te_final = np.mean(te_values[quality_fit])
        return te_final

    def compute_plane_te(self, r_squared_cut):
        te_contour = np.zeros((self.n_x, self.n_y))
        for i, pos in enumerate(self.positions):
            j = i % self.n_x
            k = i // self.n_y

            te = self.compute_characteristic_te(self.sweep_voltage[i], self.probe_current[i], r_squared_cut)

            te_contour[j, k] = te
        return te_contour

    def compute_I_sat_plane(self):
        I_sat_plane = np.zeros((self.n_x, self.n_y))
        for i, pos in enumerate(self.positions):
            j = i % self.n_x
            k = i // self.n_y

            I_sat_plane[j, k] = np.mean(self.probe_current[i][:int(0.1 * len(self.probe_current[i]))])
        return I_sat_plane

    def compute_bohm_velocity(self):
        return np.sqrt(self.te_contour / (self.amu * m_p)) * 2.99e10  # cm/s

    def compute_electron_density(self):
        v_bohm = self.compute_bohm_velocity()
        return self.I_sat_plane / (1.6e-19 * v_bohm * self.probe_area * 0.6)

    def plot_sweep(self, file_path, iter):
        f, a = plt.subplots(1, 1)

        a.plot(self.probe_time, self.sweep_voltage[iter], color='k', label='Sweep Voltage')

        a1 = a.twinx()

        a1.plot(self.probe_time, self.probe_current[iter] * 1e3, color='r', label='Probe Current')
        f.legend(loc=(0.15, 0.7))
        a.set_title("Langmuir Sweep", fontsize=15)
        a.set_xlabel(r'$t$ [$\mu s$]')
        a.set_ylabel(r'$V_p$ [$V$]')
        a1.set_ylabel(r'$I_p$ [mA]')

        f.savefig(file_path)
        plt.show()

    def plot_i_v_characteristic(self, file_path, iter):
        f, a = plt.subplots(1, 1)

        a.plot(self.sweep_voltage[iter], self.probe_current[iter] * 1e3, color='k', label='I-V Trace')

        f.legend(loc=(0.15, 0.7))
        a.set_title("Langmuir I-V Characteristc", fontsize=13)
        a.set_xlabel(r'$V_p$ [$V$]')
        a.set_ylabel(r'$I_p$ [mA]')

        f.savefig(file_path)

        plt.show()

    def plot_te_distribution(self, file_path):
        f, a = plt.subplots(1, 1)

        a.hist(self.te_contour.flatten(), bins=20)
        a.set_xlabel(r"$T_e$ [eV]")
        a.set_title(r'$T_e Distribution$', fontsize=12)
        a.set_ylabel("Counts")

        f.savefig(file_path)
        plt.show()

    def plot_te_image(self, file_path):
        f, a = plt.subplots(1, 1)

        plot_extent = [np.min(self.x_positions), np.max(self.x_positions),
                       np.min(self.y_positions), np.max(self.y_positions)]

        im = a.imshow(self.te_contour, origin='lower', extent=plot_extent, cmap='plasma')

        cbar = f.colorbar(im, label=r'$T_e$ [eV]')

        a.set_xlabel('x [cm]')
        a.set_ylabel('y [cm]')

        a.set_title("Langmuir Scan", fontsize=15)
        f.savefig(file_path)
        plt.show()

    def plot_te_contour(self, file_path, n_contour=4):

        f, a = plt.subplots(1, 1)

        plot_extent = [np.min(self.x_positions), np.max(self.x_positions),
                       np.min(self.y_positions), np.max(self.y_positions)]

        a.contour(self.te_contour,origin='lower', extent=plot_extent, colors='k', levels=3)

        a.set_xlabel('x [cm]')
        a.set_ylabel('y [cm]')

        a.set_title("Temperature Contour Map", fontsize=12)
        f.savefig(file_path)
        plt.show()


    def plot_I_sat_image(self, file_path):
        f, a = plt.subplots(1, 1)

        plot_extent = [np.min(self.x_positions), np.max(self.x_positions),
                       np.min(self.y_positions), np.max(self.y_positions)]

        im = a.imshow(self.I_sat_plane*1e3, origin='lower', extent=plot_extent, cmap='plasma')

        cbar = f.colorbar(im, label=r'$I_{is}$ [mA]')

        a.set_xlabel('x [cm]')
        a.set_ylabel('y [cm]')

        a.set_title("Langmuir Scan", fontsize=15)
        f.savefig(file_path)
        plt.show()
        pass

    def plot_n_e_image(self, file_path):
        f, a = plt.subplots(1, 1)

        plot_extent = [np.min(self.x_positions), np.max(self.x_positions),
                       np.min(self.y_positions), np.max(self.y_positions)]

        im = a.imshow(self.n_e, origin='lower', extent=plot_extent, cmap='plasma')

        cbar = f.colorbar(im, label=r'$I_{is}$ [$cm^{-3}$]')

        a.set_xlabel('x [cm]')
        a.set_ylabel('y [cm]')

        a.set_title("Langmuir Scan", fontsize=15)
        f.savefig(file_path)
        plt.show()
        pass