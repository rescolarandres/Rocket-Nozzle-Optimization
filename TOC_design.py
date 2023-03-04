import numpy as np
from table_generator import generate_table
import os
import matplotlib.pyplot as plt
from scipy.optimize import newton, fsolve
from scipy.interpolate import interp1d
from secant import secant
import math as math
import pandas as pd

def alpha(mach):
    return np.arcsin(1 / mach)


def nu_prandtl(m, gamma):
    return np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)/(gamma+1)*(m**2-1)))-np.arctan(np.sqrt(m**2-1))


def search_index(array, value):
    value_list = min(list(array), key=lambda y: abs(y - value))
    index = list(array).index(value_list)
    return index


class bell_optimized(object):
    """
       MOC procedure to obtain characteristic curves and characteristic net.
    """

    def __init__(self, folder_path, design_mach, gamma, characteristics, start_angle, init_mach, exit_angle, mid_angle,
                 exp_ratio):

        self.folder_path = folder_path
        self.design_mach = design_mach
        self.gamma = gamma
        self.characteristics = characteristics
        self.start_angle = start_angle
        self.init_mach = init_mach
        self.exit_angle = exit_angle
        self.exp_ratio = exp_ratio
        self.Pc_mean_pa = 4.107e6  # np.mean(main.Pc)
        self.T_mean_K = 3396  # stagnation temperature in chamber (given in main_propulsion)
        self.mid_angle = mid_angle
        self.R = 395.49
        self.rho0 = self.Pc_mean_pa / (self.R * self.T_mean_K)

        self.mach_numbers, self.mach_angles, self.prandtl_meyer = generate_table(self.gamma,
                                                                                 (self.init_mach, self.design_mach),
                                                                                 0.0001,
                                                                                 6,
                                                                                 write_csv=False)

    def __call__(self, *args, **kwargs):
        self.points = list(range(int(self._number_of_points(self.characteristics))))
        self.char_net_setup()

        """We compute the initial expansion characteristics and TB contour properties"""
        self.table_init = self._char_net(0, 1, self.init_mach)
        self._contour_init()
        plt.figure()
        plt.plot(self.xtb_yt, self.ytb_yt, label='Diverging section')
        plt.plot(self.table_init['x'], self.table_init['y'], 'ko', markersize=5, label='Intersecting points')
        plt.xlabel(r'$x/y_{t}$')
        plt.ylabel(r'$y/y_{t}$')
        plt.legend()
        plt.show()

        self._compute_mach()
        '''Mesh is done with the TB contour and right characteristics'''
        number_mesh_lines = 10
        # Choose what to plot: mach; char_lines
        plot = 'mach'
        self._solve_kernel_(plot, number_mesh_lines)

        '''Left char at D es computed (end control surface)'''
        self.cs_char = self._solve_control_surface()

        '''Solve transition region'''
        self.transition_table = self._solve_transition_zone()
        plt.tricontourf(self.transition_table['x'], self.transition_table['y'], self.transition_table['M'], 100, cmap='rainbow')
        contour = plt.xlabel(r' $x/y_{t}$')
        contour = plt.ylabel(r' $y/y_{t}$')
        contour = plt.colorbar()
        contour.set_label('Mach', rotation=90)
        contour = plt.axis('equal')
        plt.title('Mach contour in Transition region')
        contour = plt.show()

        """Transition characteristics plot"""
        plt.figure()
        plt.plot(self.xtb_yt, self.ytb_yt, label='Diverging section')
        plt.plot(self.table_kernel['x'], self.table_kernel['y'], 'r-', markersize=2)
        plt.plot(self.D_x_yt, self.D_y_yt, 'ro', markersize=5, label='Kernel limit')
        plt.plot(self.cs_char['x'], self.cs_char['y'], 'go', markersize=5, label='Control surface')
        plt.plot(self.transition_table['x'], self.transition_table['y'], 'ko', markersize=2, label='Intersecting points')
        plt.axis('equal')
        plt.legend()
        plt.xlabel(r'$x/y_{t}$')
        plt.ylabel(r'$y/y_{t}$')
        plt.show()

        '''Nozzle contour'''
        self.midcontour_table = self._solve_contour()
        plt.figure()
        plt.plot(self.transition_table['x'], self.transition_table['y'], 'ko', markersize=1)
        plt.plot(self.midcontour_table['x'], self.midcontour_table['y'], 'k-', label='Nozzle contour')
        plt.axis('equal')
        plt.legend()
        plt.xlabel(r'$x/y_{t}$')
        plt.ylabel(r'$y/y_{t}$')
        plt.axis('equal')
        plt.show()

        contour_x = np.append(self.diverging_contour['x'], self.midcontour_table['x'])
        contour_y = np.append(self.diverging_contour['y'], self.midcontour_table['y'])
        new_x = np.linspace(contour_x[0], contour_x[-1], 500)
        y = interp1d(contour_x, contour_y, kind='cubic')

        self.contour_table = np.zeros((500, 2))
        self.contour_table[:, 0] = new_x
        self.contour_table[:, 1] = y(new_x)
        plt.plot(self.contour_table[:, 0], self.contour_table[:, 1], label='Wall contour')
        plt.xlabel(r'$x/y_{t}$')
        plt.ylabel(r'$y/y_{t}$')
        plt.legend()
        plt.axis('equal')
        plt.show()

        '''Generate csv'''
        mat = np.matrix(self.contour_table)
        df = pd.DataFrame(data=mat.astype(float))
        df.to_csv('outfile.csv', sep=';', header=False, float_format='%.2f', index=False)

    def _solve_contour(self):
        y0 = self.ytb_yt[-1]
        x0 = self.xtb_yt[-1]
        theta = self.mid_angle
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y',
                                  'u', 'v']
        contour_index = []
        diff = 10
        for i in range(0, self.index-1):
            for j in range(0, len(self.cs_char)-1):
                k = (self.index*j) + i
                x = self.transition_table['x'][k]
                y = y0+(x-x0)*np.tan(theta)
                error = abs(y - self.transition_table['y'][k])
                if i == 0 and error < diff:
                    diff = error
                    index = k
                elif i != 0 and error < diff:
                    diff = error
                    index = (self.index*(j-2)) + i

            contour_index.append(index)
            y0 = self.transition_table['y'][index]
            x0 = self.transition_table['x'][index]
            theta = self.transition_table['theta'][index]
            diff = 10

        self.moc_table = np.zeros(len(contour_index)+1, dtype=[(header, 'f') for header in self.moc_table_headers])
        index_remove = []
        for i in range(0, len(contour_index)):
            self.moc_table['R-'][i] = self.transition_table['R-'][contour_index[i]]
            self.moc_table['R+'][i] = self.transition_table['R+'][contour_index[i]]
            self.moc_table['theta'][i] = self.transition_table['theta'][contour_index[i]]
            self.moc_table['nu'][i] = self.transition_table['nu'][contour_index[i]]
            self.moc_table['M'][i] = self.transition_table['M'][contour_index[i]]
            self.moc_table['m_angle'][i] = self.transition_table['m_angle'][contour_index[i]]
            self.moc_table['theta+m_angle'][i] = self.transition_table['theta+m_angle'][contour_index[i]]
            self.moc_table['theta-m_angle'][i] = self.transition_table['theta-m_angle'][contour_index[i]]
            self.moc_table['u'][i] = self.transition_table['u'][contour_index[i]]
            self.moc_table['v'][i] = self.transition_table['v'][contour_index[i]]
            self.moc_table['x'][i] = self.transition_table['x'][contour_index[i]]
            self.moc_table['y'][i] = self.transition_table['y'][contour_index[i]]
            if self.moc_table['y'][i] < self.moc_table['y'][i-1]:
                index_remove.append(i)
        self.moc_table = np.delete(self.moc_table, index_remove)

        # We add last point from the control surface
        i = -1
        self.moc_table['R-'][i] = self.cs_char['R-'][i]
        self.moc_table['R+'][i] = self.cs_char['R+'][i]
        self.moc_table['theta'][i] = self.cs_char['theta'][i]
        self.moc_table['nu'][i] = self.cs_char['nu'][i]
        self.moc_table['M'][i] = self.cs_char['M'][i]
        self.moc_table['m_angle'][i] = self.cs_char['m_angle'][i]
        self.moc_table['theta+m_angle'][i] = self.cs_char['theta+m_angle'][i]
        self.moc_table['theta-m_angle'][i] = self.cs_char['theta-m_angle'][i]
        self.moc_table['u'][i] = self.moc_table['M'][i] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][i] ** 2)) * np.cos(self.moc_table['theta'][i])
        self.moc_table['v'][i] = self.moc_table['M'][i] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][i] ** 2)) * np.sin(self.moc_table['theta'][i])
        self.moc_table['x'][i] = self.cs_char['x'][i]
        self.moc_table['y'][i] = self.cs_char['y'][i]
        return self.moc_table

    def _solve_transition_zone(self):
        self.index = search_index(self.table_kernel['x'], self.D_x_yt)
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y','u','v']
        self.moc_table = np.zeros(len(self.cs_char)*self.index, dtype=[(header, 'f') for header in self.moc_table_headers])

        for i in range(0, len(self.cs_char)):

            for j in range(self.index, 0, -1):

                k = j+(self.index*i)-1
                l = j+(self.index*(i-1))-1

                if j == self.index:
                    self.moc_table['R-'][k] = self.cs_char['R-'][i]
                    self.moc_table['R+'][k] = self.cs_char['R+'][i]
                    self.moc_table['theta'][k] = self.cs_char['theta'][i]
                    self.moc_table['nu'][k] = self.cs_char['nu'][i]
                    self.moc_table['M'][k] = self.cs_char['M'][i]
                    self.moc_table['m_angle'][k] = self.cs_char['m_angle'][i]
                    self.moc_table['theta+m_angle'][k] = self.moc_table['theta'][k] + self.moc_table['m_angle'][k]
                    self.moc_table['theta-m_angle'][k] = self.moc_table['theta'][k] - self.moc_table['m_angle'][k]
                    self.moc_table['u'][k] = self.moc_table['M'][k] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.cos(self.moc_table['theta'][k])
                    self.moc_table['v'][k] = self.moc_table['M'][k] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.sin(self.moc_table['theta'][k])
                    self.moc_table['x'][k] = self.cs_char['x'][i]
                    self.moc_table['y'][k] = self.cs_char['y'][i]

                else:

                    def get_xy(y_plus, x_plus, t_plus, t_minus, y_minus, x_minus):
                        x = (y_minus - y_plus + x_plus * t_plus - x_minus * t_minus) / (t_plus - t_minus)
                        y = y_minus + (x - x_minus) * t_minus
                        return x, y

                    if i == 0:
                        self.moc_table['R-'][k] = self.table_kernel['R-'][j]
                        self.moc_table['R+'][k] = self.table_kernel['R+'][j]
                        self.moc_table['theta'][k] = self.table_kernel['theta'][j]
                        self.moc_table['nu'][k] = self.table_kernel['nu'][j]
                        self.moc_table['M'][k] = self.table_kernel['M'][j]
                        self.moc_table['m_angle'][k] = self.table_kernel['m_angle'][j]
                        self.moc_table['theta+m_angle'][k] = self.moc_table['theta'][k] + self.moc_table['m_angle'][k]
                        self.moc_table['theta-m_angle'][k] = self.moc_table['theta'][k] - self.moc_table['m_angle'][k]
                        self.moc_table['u'][k] = self.moc_table['M'][k] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.cos(self.moc_table['theta'][k])
                        self.moc_table['v'][k] = self.moc_table['M'][k] * np.sqrt(self.gamma * self.R * self.T_mean_K / (1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.sin(self.moc_table['theta'][k])
                        self.moc_table['x'][k] = self.table_kernel['x'][j]
                        self.moc_table['y'][k] = self.table_kernel['y'][j]
                    else:
                        self.moc_table['R-'][k] = self.cs_char['R-'][i]
                        self.moc_table['R+'][k] = self.table_kernel['R+'][j]
                        self.moc_table['theta'][k] = 0.5*(self.cs_char['R-'][i]+self.moc_table['R+'][k])
                        self.moc_table['nu'][k] = 0.5*(self.moc_table['R-'][k]-self.moc_table['R+'][k])
                        self.moc_table['M'][k] = self.mach_numbers[search_index(self.prandtl_meyer, np.rad2deg(self.moc_table['nu'][k]))]
                        self.moc_table['m_angle'][k] = alpha(self.moc_table['M'][k])
                        self.moc_table['theta+m_angle'][k] = self.moc_table['theta'][k] + self.moc_table['m_angle'][k]
                        self.moc_table['theta-m_angle'][k] = self.moc_table['theta'][k] - self.moc_table['m_angle'][k]
                        self.moc_table['u'][k] = self.moc_table['M'][k] * np.sqrt(
                            self.gamma * self.R * self.T_mean_K / (
                                        1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.cos(
                            self.moc_table['theta'][k])
                        self.moc_table['v'][k] = self.moc_table['M'][k] * np.sqrt(
                            self.gamma * self.R * self.T_mean_K / (
                                        1 + 0.5 * (self.gamma - 1) * self.moc_table['M'][k] ** 2)) * np.sin(
                            self.moc_table['theta'][k])
                        x0 = self.moc_table['x'][l]
                        y0 = self.moc_table['y'][l]
                        t0 = self.moc_table['theta+m_angle'][l]

                        x, y = get_xy(y0, x0, np.tan(t0), np.tan(self.moc_table['theta-m_angle'][k + 1]),
                                      self.moc_table['y'][k + 1], self.moc_table['x'][k + 1])
                        # if y < self.table_kernel['y'][j]:
                        #     self.moc_table['x'][k] = self.table_kernel['x'][j]
                        #     self.moc_table['y'][k] = self.table_kernel['y'][j]
                        # else:
                        self.moc_table['x'][k] = x
                        self.moc_table['y'][k] = y

        return self.moc_table

    def _solve_control_surface(self):
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.moc_table = np.zeros((len(self.y_ye) - search_index(self.y_ye, self.D_y_ye)), dtype=[(header, 'f') for header in self.moc_table_headers])
        x0 = self.D_x_yt
        y0 = self.D_y_yt
        for i in range(0, len(self.moc_table)):
            idx_cs = i + search_index(self.y_ye, self.D_y_ye)
            self.moc_table['R-'][i] = -self.theta_cs[idx_cs]+nu_prandtl(self.mach_cs[idx_cs], self.gamma)
            self.moc_table['R+'][i] = self.theta_cs[-1]-nu_prandtl(self.mach_cs[idx_cs], self.gamma)
            self.moc_table['theta'][i] = self.theta_cs[idx_cs]
            self.moc_table['nu'][i] = np.deg2rad(self.prandtl_meyer[search_index(self.mach_numbers, self.mach_cs[idx_cs])])
            self.moc_table['M'][i] = self.mach_cs[idx_cs]
            self.moc_table['m_angle'][i] = float(alpha(self.moc_table['M'][i]))
            self.moc_table['theta+m_angle'][i] = self.moc_table['theta'][i] + self.moc_table['m_angle'][i]
            self.moc_table['theta-m_angle'][i] = self.moc_table['theta'][i] - self.moc_table['m_angle'][i]
            self.moc_table['y'][i] = self.y_ye[idx_cs]*np.sqrt(self.exp_ratio)
            self.moc_table['x'][i] = (self.moc_table['y'][i]-y0)/np.tan(self.theta_cs[idx_cs]+alpha(self.mach_cs[idx_cs]))+x0
            y0 = self.moc_table['y'][i]
            x0 = self.moc_table['x'][i]

        return self.moc_table

    def _solve_kernel_(self, plot, number_mesh_lines):
        table_prior = self._wall_points(self.theta_tb[180], self.mach_tb[0, 180], self.xtb_yt[180], self.ytb_yt[180],
                                        self.table_init)
        if plot == 'char_lines':
            grid = plt.figure(1)
            for i in np.linspace(200, len(self.xtb_yt) - 10, number_mesh_lines):
                i = math.floor(i)
                grid = plt.plot(table_prior['x'], table_prior['y'], 'k-', markersize=3)

                table_mesh = self._mesh_points(self.theta_tb[i - 1], self.mach_tb[0, i - 1], self.xtb_yt[i],
                                               self.ytb_yt[i], table_prior)
                table_prior = table_mesh

            ''' Compute the right char line which will be Kernel  '''
            self.table_kernel = self._mesh_points(self.theta_tb[-1], self.mach_tb[0, -1], self.xtb_yt[-1], self.ytb_yt[-1],
                                              table_prior)

            '''We obtain point D of Kernel extension by continuity and eqs of the control surface'''
            self._control_surface()
            self._kernel_extension()
            grid = plt.plot(table_prior['x'], table_prior['y'], 'k-', markersize=3, label='Characteristic lines')
            grid = plt.plot(self.table_init['x'], self.table_init['y'], 'ko', markersize=3, label='Intersecting points')
            grid = plt.plot(self.xtb_yt, self.ytb_yt, label='Diverging section')
            grid = plt.plot(self.table_kernel['x'], self.table_kernel['y'], 'r-', markersize=3, label='Kernel limit')
            grid = plt.plot(self.D_x_yt, self.D_y_yt, 'ro', markersize=5, label='Kernel end point')
            grid = plt.axis('equal')
            plt.legend()
            plt.ylim([0, 2])
            grid = plt.xlabel(r'$x/y_{t}$')
            grid = plt.ylabel(r'$y/y_{t}$')
            plt.show()
        elif plot == 'mach':

            for i in np.linspace(180, len(self.xtb_yt) - 3, number_mesh_lines):
                i = math.floor(i)
                table_mesh = self._mesh_points(self.theta_tb[i - 1], self.mach_tb[0, i - 1], self.xtb_yt[i],
                                               self.ytb_yt[i]
                                               , table_prior)
                table_prior = table_mesh
            ''' Compute the right char line which will be Kernel  '''
            self.table_kernel = self._mesh_points(self.theta_tb[-1], self.mach_tb[0, -1], self.xtb_yt[-1],
                                                  self.ytb_yt[-1], table_prior)
            contour = plt.figure(2)
            contour = plt.tricontourf(np.concatenate((self.table_init['x'][:],self.table_kernel['x'][:])), np.concatenate((self.table_init['y'][:],self.table_kernel['y'][:])), np.concatenate((self.table_init['M'][:],self.table_kernel['M'][:])), 100, cmap='rainbow')
            contour = plt.xlabel(r' $x/y_{t}$')
            contour = plt.ylabel(r' $y/y_{t}$')
            contour = plt.colorbar()
            contour.set_label('Mach', rotation=90)
            contour = plt.axis('equal')
            plt.title('Mach contour in Kernel region')
            contour = plt.show()

            '''We obtain point D of Kernel extension by continuity and eqs of the control surface'''
            self._control_surface()
            self._kernel_extension()

        # Plotting Control surface properties
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(self.y_ye, np.rad2deg(self.theta_cs))
        ax1.plot(self.D_y_ye, np.rad2deg(self.theta_D), 'ro', label='Kernel end point')
        ax1.set_title("Angle between flow and nozzle locus")
        ax1.set_xlabel(r' $y/y_{e}$')
        ax1.set_ylabel(r' $\theta [deg]$')
        ax2.plot(self.y_ye, self.mach_cs, )
        ax2.plot(self.D_y_ye, self.mach_D, 'ro', label='Kernel end point')
        ax2.set_title("Mach locus")
        ax2.set_xlabel(r' $y/y_{e}$')
        ax2.set_ylabel(r' $Mach$')
        fig.tight_layout()
        plt.legend()
        plt.show()

    def _number_of_points(self, n):
        """
        For symmetric planar nozzle only
        :param n: integer number of characteristic lines
        :return: integer number of points from those lines
        """
        if n == 0:
            return 0
        elif n == 1:
            return 3
        elif n == 2:
            return 2 * 3 + 1
        else:
            return n*(n+1)/2

    def char_net_setup(self):
        """
        This function obtains the initial sweep of characteristic lines until they reach nozzle axis by sweeping theta
        """

        # Calculate theta max
        theta_max = 90-(self.prandtl_meyer[self.mach_numbers.index(self.design_mach)] / 2.0)

        self.delta_theta = (np.deg2rad(theta_max)-self.start_angle)/self.characteristics

    def _char_net(self, x0, y0, mach_init):
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.moc_table = np.zeros(len(self.points), dtype=[(header, 'f') for header in self.moc_table_headers])

        def f(machs, nu):
            return nu_prandtl(machs, self.gamma)-nu

        R_plus = np.ones((1, self.characteristics))
        R_minus = np.ones((1, self.characteristics))

        for idx in range(0, self.characteristics):
            R_minus[0, idx] = abs((idx)*self.delta_theta + nu_prandtl(mach_init, self.gamma))

        for idx in range(1, self.characteristics):
            R_plus[0, idx] = -R_minus[0, idx]

        R_plus[0, 0] = -np.deg2rad(0.375)
        R_minus[0, 0] = np.deg2rad(0.375)
        sum = 0
        row = 0

        for i in range(0, self.characteristics):

            for j in range(0, (self.characteristics-i)):
                if i == 0:
                    self.moc_table['theta'][j] = 0
                    self.moc_table['nu'][j] = R_minus[0, j]
                    self.moc_table['M'][j] = float(newton(f, x0=1.1, args=(self.moc_table['nu'][j],)))
                    self.moc_table['m_angle'][j] = float(alpha(self.moc_table['M'][j]))
                    self.moc_table['R-'][j] = R_minus[0, j]
                    self.moc_table['R+'][j] = R_plus[0, j]
                    self.moc_table['theta+m_angle'][j] = np.deg2rad(self.moc_table['theta'][j]) + self.moc_table['m_angle'][j]
                    self.moc_table['theta-m_angle'][j] = np.deg2rad(self.moc_table['theta'][j]) - self.moc_table['m_angle'][j]
                    self.moc_table['x'][j] = np.tan(self.delta_theta*(j+1))
                    self.moc_table['y'][j] = 0

                else:
                    k = j+sum
                    l = row
                    self.moc_table['R-'][k] = R_minus[0, j + i]
                    self.moc_table['R+'][k] = R_plus[0, j]
                    self.moc_table['theta'][k] = 0.5*(self.moc_table['R-'][k]+self.moc_table['R+'][k])
                    self.moc_table['nu'][k] = 0.5*(self.moc_table['R-'][k]-self.moc_table['R+'][k])
                    self.moc_table['M'][k] = float(newton(f, x0=1.1, args=(self.moc_table['nu'][k],)))
                    self.moc_table['m_angle'][k] = float(alpha(self.moc_table['M'][k]))
                    self.moc_table['theta+m_angle'][k] = self.moc_table['theta'][k] + self.moc_table['m_angle'][k]
                    self.moc_table['theta-m_angle'][k] = self.moc_table['theta'][k] - self.moc_table['m_angle'][k]

                    def get_xy(y_bar, x_bar, t_plus, t_minus):
                        x = (y0-y_bar+x_bar*t_plus-x0*t_minus)/(t_plus+t_minus)
                        y = y0-(x-x0)*t_minus
                        return x, y
                    # Get x and y coordinates of point intersecting two char lines
                    self.moc_table['x'][k], self.moc_table['y'][k] = get_xy(self.moc_table['y'][l+j],
                                                                            self.moc_table['x'][l+j],
                                                                            np.tan(self.moc_table['theta+m_angle'][l+j]),
                                                                            np.tan(np.pi/2-(j+1+i)*self.delta_theta))

            if sum == self.characteristics+1:
                row = 0
            else:
                row = sum
            sum = sum + j + 1

        return self.moc_table

    def _contour_init(self):
        """AT contour"""
        thetaat = np.deg2rad(np.linspace(-135, -90, 50))  # 135 Typical contraction angle
        self.xat_yt = 1.5 * np.cos(thetaat)
        self.yat_yt = 1.5 * np.sin(thetaat) + 1.5 + 1

        """TB contour"""
        thetatb = np.deg2rad(np.linspace(-90, np.rad2deg(self.mid_angle) - 90, 400))
        self.xtb_yt = 0.45 * np.cos(thetatb)
        self.ytb_yt = 0.45 * np.sin(thetatb) + 0.45 + 1
        self.theta_tb = np.arctan(np.asarray(np.diff(self.ytb_yt) / np.diff(self.xtb_yt)))

        moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.diverging_contour = np.zeros(len(self.xtb_yt), dtype=[(header, 'f') for header in moc_table_headers])
        # self.diverging_contour['x'] = np.append(self.xat_yt, self.xtb_yt[:])
        # self.diverging_contour['y'] = np.append(self.yat_yt, self.ytb_yt[:])
        self.diverging_contour['x'] = self.xtb_yt[:]
        self.diverging_contour['y'] = self.ytb_yt[:]

    def _kernel_extension(self):
        def shoot(d_x):
            indexx = search_index(self.table_kernel['x'][:], d_x)
            y_aux = self.table_kernel['y'][indexx] / np.sqrt(self.exp_ratio)
            indexy = search_index(self.y_ye, y_aux)

            def rho_rhot(mach):
                return (1+0.5*(self.gamma-1)*1.103**2)**(1/(self.gamma-1)) / \
                       (1+0.5*(self.gamma-1)*mach**2)**(1/(self.gamma-1))

            def W_Wt(mach):
                return mach*np.sqrt((1+0.5*(self.gamma-1)*1.103**2)/(1+0.5*(self.gamma-1)*mach**2))

            def int1(dx_yt, y, mach, theta):
                return rho_rhot(mach) * W_Wt(mach) * np.sin(alpha(mach)) / (np.cos(theta - alpha(mach))) * y * dx_yt

            def int2(dy_ye, mach, theta, y_ye):
                return rho_rhot(mach) * W_Wt(mach) * np.sin(alpha(mach)) / (np.sin(theta + alpha(mach))) * y_ye * dy_ye

            integral1 = 0
            for i in range(0, indexx):
                dx = self.table_kernel['x'][i+1] - self.table_kernel['x'][i]
                integral1 = integral1 + int1(dx, self.table_kernel['y'][i], self.table_kernel['M'][i],
                                             self.table_kernel['theta'][i])

            integral2 = 0
            dy = abs(self.y_ye[1] - self.y_ye[0])
            for j in range(indexy, len(self.mach_cs)):
                integral2 = integral2 + int2(dy, self.mach_cs[j], self.theta_cs[j], self.y_ye[j])

            f = integral1 - self.exp_ratio * integral2
            return f

        D_x_yt = 4.5 #secant(shoot, 1, 5, 100)
        self.D_x_yt = self.table_kernel['x'][search_index(self.table_kernel['x'][:], D_x_yt)]
        self.D_y_yt = self.table_kernel['y'][search_index(self.table_kernel['x'][:], D_x_yt)]
        D_y_ye = self.D_y_yt / np.sqrt(self.exp_ratio)
        self.D_y_ye = self.y_ye[search_index(self.y_ye, D_y_ye)]
        self.mach_D = self.mach_cs[search_index(self.y_ye, self.D_y_ye)]
        self.theta_D = self.theta_cs[search_index(self.y_ye, self.D_y_ye)]

    def _control_surface(self):
        """Find variables in the control surface (M,theta)"""
        mach_asth = lambda mach: np.sqrt(1 / (self.gamma - 1 + 2 / mach ** 2))
        self.y_ye = np.linspace(0.2, 1, 50)

        def fun(x, y_ye):
            return [mach_asth(x[0]) * np.cos(x[1] - alpha(x[0])) / np.cos(alpha(x[0])) -
                    mach_asth(self.design_mach) * np.cos(self.exit_angle - alpha(self.design_mach)) /
                    np.cos(alpha(self.design_mach)),
                    y_ye * x[0] ** 2 * (1 + 0.5 * (self.gamma - 1) * x[0] ** 2) ** (-self.gamma / (self.gamma - 1)) * (
                    np.sin(x[1])) ** 2 * np.tan(alpha(x[0])) - self.design_mach ** 2 * (1 + 0.5 * (self.gamma - 1) *
                    self.design_mach ** 2) ** (-self.gamma / (self.gamma - 1)) * (np.sin(self.exit_angle)) ** 2 *
                    np.tan(alpha(self.design_mach))]

        roots = [fsolve(fun, x0=np.asarray([3.5, 0.2]), args=(i,)) for i in self.y_ye]
        self.mach_cs = np.asarray([roots[j][0] for j in range(0, len(self.y_ye))])
        self.theta_cs = np.asarray([roots[j][1] for j in range(0, len(self.y_ye))])

    def _compute_mach(self):
        """Function that computes the mach number given a TB geometry"""
        self.mach_tb = np.zeros((1, len(self.xtb_yt)-1))
        self.mach_tb[0, 0] = self.init_mach
        mach1 = self.mach_tb[0, 0]
        for j in range(1, len(self.xtb_yt) - 1):
            nu_mach1 = nu_prandtl(mach1, self.gamma)
            nu_mach2 = self.theta_tb[j] - self.theta_tb[j - 1] + nu_mach1  # Angle change of flow in nozzle (concave)
            idx = search_index(self.prandtl_meyer, np.rad2deg(nu_mach2))
            mach2 = self.mach_numbers[idx]
            mach1 = float(mach2)
            self.mach_tb[0, j] = mach2

    def _wall_points(self, theta_init, mach_init, x_init, y_init, table_prior):
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.moc_table = np.zeros(self.characteristics+1, dtype=[(header, 'f') for header in self.moc_table_headers])

        def f(machs, nu):
            return nu_prandtl(machs, self.gamma)-nu

        sum = len(table_prior) - 1   # As we begin now from the top, we have to use points to start the sequence
        for j in range(0, self.characteristics+1):
            if j == 0:
                """Initial wall point"""
                self.moc_table['theta'][j] = theta_init
                idx = search_index(self.mach_numbers, mach_init)
                self.moc_table['nu'][j] = np.deg2rad(self.prandtl_meyer[idx])
                self.moc_table['M'][j] = self.mach_numbers[idx]
                self.moc_table['m_angle'][j] = np.deg2rad(self.mach_angles[idx])
                self.moc_table['R-'][j] = self.moc_table['theta'][j]+self.moc_table['nu'][j]
                self.moc_table['R+'][j] = self.moc_table['theta'][j]-self.moc_table['nu'][j]
                self.moc_table['theta+m_angle'][j] = self.moc_table['theta'][j] + self.moc_table['m_angle'][j]
                self.moc_table['theta-m_angle'][j] = self.moc_table['theta'][j] - self.moc_table['m_angle'][j]
                self.moc_table['x'][j] = x_init
                self.moc_table['y'][j] = y_init

            else:       # Condition is internal point
                k = sum
                self.moc_table['theta'][j] = 0.5 * (self.moc_table['R-'][0] + table_prior['R+'][k])
                self.moc_table['nu'][j] = 0.5 * (self.moc_table['R-'][0] - table_prior['R+'][k])
                self.moc_table['M'][j] = float(newton(f, x0=1.1, args=(abs(self.moc_table['nu'][j]),)))
                self.moc_table['m_angle'][j] = float(alpha(self.moc_table['M'][j]))
                self.moc_table['R-'][j] = self.moc_table['R-'][0]
                self.moc_table['R+'][j] = table_prior['R+'][k]
                self.moc_table['theta+m_angle'][j] = self.moc_table['theta'][j] + self.moc_table['m_angle'][j]
                self.moc_table['theta-m_angle'][j] = self.moc_table['theta'][j] - self.moc_table['m_angle'][j]

                def get_xy(y_plus, x_plus, t_plus, t_minus, y_minus, x_minus):
                    x = (y_minus-y_plus+x_plus*t_plus-x_minus*t_minus)/(t_plus-t_minus)
                    y = y_minus + (x - x_minus) * t_minus
                    return x, y
                    # Get x and y coordinates of point intersecting two char lines

                self.moc_table['x'][j], self.moc_table['y'][j] = get_xy(table_prior['y'][k],
                                                                        table_prior['x'][k],
                                                                        np.tan(table_prior['theta+m_angle'][k]),
                                                                        np.tan(self.moc_table['theta-m_angle'][j-1]),
                                                                        self.moc_table['y'][j-1],
                                                                        self.moc_table['x'][j-1])

                sum = sum - j
        return self.moc_table

    def _mesh_points(self, theta_init, mach_init, x_init, y_init, table_prior):
        self.moc_table_headers = ['R+', 'R-', 'theta', 'nu', 'M', 'm_angle', 'theta+m_angle', 'theta-m_angle', 'x', 'y']
        self.moc_table = np.zeros(self.characteristics + 1, dtype=[(header, 'f') for header in self.moc_table_headers])

        def f(machs, nu):
            return nu_prandtl(machs, self.gamma) - nu

        for j in range(0, self.characteristics + 1):
            if j == 0:
                """Initial wall point"""
                self.moc_table['theta'][j] = theta_init
                idx = search_index(self.mach_numbers, mach_init)
                self.moc_table['nu'][j] = np.deg2rad(self.prandtl_meyer[idx])
                self.moc_table['M'][j] = self.mach_numbers[idx]
                self.moc_table['m_angle'][j] = np.deg2rad(self.mach_angles[idx])
                self.moc_table['R-'][j] = self.moc_table['theta'][j] + self.moc_table['nu'][j]
                self.moc_table['R+'][j] = -self.moc_table['theta'][j] + self.moc_table['nu'][j]
                self.moc_table['theta+m_angle'][j] = self.moc_table['theta'][j] + self.moc_table['m_angle'][j]
                self.moc_table['theta-m_angle'][j] = self.moc_table['theta'][j] - self.moc_table['m_angle'][j]
                self.moc_table['x'][j] = x_init
                self.moc_table['y'][j] = y_init

            else:  # Condition is internal point

                self.moc_table['theta'][j] = 0.5 * (self.moc_table['R-'][0] + table_prior['R+'][j])
                self.moc_table['nu'][j] = 0.5 * (self.moc_table['R-'][0] - table_prior['R+'][j])
                self.moc_table['M'][j] = float(newton(f, x0=1.1, args=(abs(self.moc_table['nu'][j]),)))
                self.moc_table['m_angle'][j] = float(alpha(self.moc_table['M'][j]))
                self.moc_table['R-'][j] = self.moc_table['R-'][0]
                self.moc_table['R+'][j] = table_prior['R+'][j]
                self.moc_table['theta+m_angle'][j] = self.moc_table['theta'][j] + self.moc_table['m_angle'][j]
                self.moc_table['theta-m_angle'][j] = self.moc_table['theta'][j] - self.moc_table['m_angle'][j]

                def get_xy(y_plus, x_plus, t_plus, t_minus, y_minus, x_minus):
                    x = (y_minus - y_plus + x_plus * t_plus - x_minus * t_minus) / (t_plus - t_minus)
                    y = y_minus + (x - x_minus) * t_minus
                    return x, y
                    # Get x and y coordinates of point intersecting two char lines

                self.moc_table['x'][j], self.moc_table['y'][j] = get_xy(table_prior['y'][j],
                                                                        table_prior['x'][j],
                                                                        np.tan(table_prior['theta+m_angle'][j]),
                                                                        np.tan(self.moc_table['theta-m_angle'][j - 1]),
                                                                        self.moc_table['y'][j - 1],
                                                                        self.moc_table['x'][j - 1])

        return self.moc_table


if __name__ == '__main__':
    input_folder_path = os.path.dirname(os.path.realpath(__file__))
    input_design_mach = float(3.56)
    input_gamma = float(1.2)
    input_characteristics = int(10)
    input_start_angle = float(np.deg2rad(0.1))
    input_mach_init = 1.103   # Mach at throat
    input_exit_angle = float(np.deg2rad(9.96))     # angle theta E
    input_mid_angle = float(np.deg2rad(36.2686))    # angle theta N
    input_exp_ratio = float(12.382)     # Exp ratio from script
    nozzle_design = bell_optimized(input_folder_path, input_design_mach, input_gamma, input_characteristics,
                                   input_start_angle, input_mach_init, input_exit_angle, input_mid_angle,
                                   input_exp_ratio)
    nozzle_design()



