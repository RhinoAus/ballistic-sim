import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import os
import csv


class Sim:
    def __init__(self, v=100, x=0, zero_range=10, y0=1.5, rho=1.225, Cd=0.465, radius=0.006, mass=0.001, g=-9.81, time_step=0.0001, time_max=2):
        """
        Initialize the simulation parameters.

        :param v: Measured velocity (m/s)
        :param x: Distance of the measurement (m)
        :param zero_range: Distance that the shot is to be zeroed at (m). Default is 10 m.
        :param y0: Initial y position (m)
        :param time_step: Time step for the simulation (s)
        :param time_max: Maximum run time for the simulation (s)
        :param rho: Air density (kg/m^3)
        :param Cd: Drag coefficient
        :param radius: Radius of the projectile (m)
        :param mass: Mass of the projectile (kg)
        :param g: Acceleration due to gravity (m/s^2)
        """
        self.v0 = v
        self.rho = rho
        self.Cd = Cd
        self.A = np.pi * radius ** 2
        self.mass = mass
        self.g = g
        self.theta = 0
        self.time_step = time_step
        self.time_max = time_max

        self.iterations = int(np.floor(time_max / time_step)) + 1
        self.time = np.arange(0, time_max + time_step, time_step)
        self.x = np.zeros(self.iterations)
        self.y = y0 * np.ones(self.iterations)
        self.vx = np.zeros(self.iterations)
        self.vy = np.zeros(self.iterations)
        self.ax = np.zeros(self.iterations)
        self.ay = np.zeros(self.iterations)

        # Update the muzzle velocity if not measured at the muzzle
        if x > 0:
            for _ in range(3):
                self.update_v0(v, x)

        # Zero the shot
        if zero_range > 0:
            for _ in range(3):
                self.update_zero(zero_range)

    def run(self, theta=None):
        """
        Run the simulation.

        :param theta: Angle of the projectile (rad)
        """
        if theta is None:
            theta = self.theta

        self.vx[0] = self.v0 * np.cos(theta)
        self.vy[0] = self.v0 * np.sin(theta)

        for step in range(1, self.iterations):
            # Claculate the drag force
            v_total = np.sqrt(self.vx[step - 1] ** 2 + self.vy[step - 1] ** 2)
            fd_x = -0.5 * self.rho * self.Cd * self.A * self.vx[step - 1] * v_total
            fd_y = -0.5 * self.rho * self.Cd * self.A * self.vy[step - 1] * v_total

            # a = F/m
            self.ax[step - 1] = fd_x / self.mass
            self.ay[step - 1] = self.g + fd_y / self.mass

            # v = u + at
            self.vx[step] = self.vx[step - 1] + self.ax[step - 1] * self.time_step
            self.vy[step] = self.vy[step - 1] + self.ay[step - 1] * self.time_step

            # s = ut + 1/2at^2
            self.x[step] = self.x[step - 1] + self.vx[step - 1] * self.time_step + 0.5 * self.ax[step - 1] * self.time_step ** 2
            self.y[step] = self.y[step - 1] + self.vy[step - 1] * self.time_step + 0.5 * self.ay[step - 1] * self.time_step ** 2

        # capture special case of the flat shot for drop percentage comparison (to avoid 0 crossing and thus divide by zero error)
        if theta == 0:
            self.x_flat = self.x
            self.y_flat = self.y
            self.vx_flat = self.vx
            self.vy_flat = self.vy
            self.ax_flat = self.ax
            self.ay_flat = self.ay

        # magnitude of velocity
        self.v = np.sqrt(self.vx ** 2 + self.vy ** 2)

        # ke = 1/2mv^2
        self.ke = 0.5 * self.mass * self.v ** 2

    def update_zero(self, x_zero):
        """
        Calculate the zero angle for the given zeroing distance. Quite accurate after a single iteration.

        :param x_zero: Distance that the shot is to be zeroed at (m).
        """
        self.run()
        drop = np.interp(x_zero, self.x, self.y) - self.y[0]
        self.theta += np.arctan2(-drop, x_zero)

    def update_v0(self, v, x):
        """
        Update the initial velocity based on a measured velocity at a specific distance. Quite accurate after a single iteration.

        :param v: Measured velocity (m/s)
        :param x: Distance at which the velocity is measured (m)
        """
        self.run()
        v_measured = np.interp(x, self.x, self.v)
        self.v0 *= v / v_measured

    def save_to_file(self, filename):
        """
        Save the simulation data to a file.

        :param filename: Name of the file to save the data
        """
        np.savetxt(filename, np.c_[self.time, self.x, self.y, self.vx, self.vy, self.ax, self.ay])


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Nerf Dart Ballistics Simulator")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Initialize misc variables
        self.simulations = []
        self.cursor = None
        self.fields = 0
        self.colour_table = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf"   # Cyan
        ]

        # Create tabs
        notebook = ttk.Notebook(root)
        notebook.pack(expand=True, fill='both')
        notebook.rowconfigure(0, weight=1)
        notebook.columnconfigure(0, weight=1)

        # Tab 1: Run Simulations
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text='Run Simulations')

        self.velocity_fps_entry = self.create_input_field(tab1, "Velocity (ft/s):", 300)
        self.ke_entry = self.create_input_field(tab1, "Ke (J):", '')
        self.velocity_distance_m_entry = self.create_input_field(tab1, "Velocity distance (m):", 0)
        self.mass_g_entry = self.create_input_field(tab1, "Mass (g):", 1.0)
        self.drag_coefficient_entry = self.create_input_field(tab1, "Drag Coefficient:", 0.465)
        self.zero_distance_entry = self.create_input_field(tab1, "Zero Distance (m):", 10)
        self.sim_id = 1
        tk.Button(tab1, text="Run Simulation", command=self.run_simulation).grid(row=self.fields, column=0, columnspan=2)
        self.fields += 1
        tk.Button(tab1, text="Save Results", command=self.save_results).grid(row=self.fields, column=0, columnspan=2)
        self.fields += 1

        tab1.rowconfigure(self.fields + 1, weight=1)
        tab1.columnconfigure(1, weight=1)

        # Tab 2: Visualize Results
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text='Visualize Results')

        self.sim_select_frame = ttk.LabelFrame(tab2, text="Select Simulations")
        self.sim_select_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.create_plot_area(tab2)
        self.create_plot_options(tab2)

        tab2.rowconfigure(0, weight=1)
        tab2.columnconfigure(1, weight=1)

        # Tab 3: Back-calculate Cd
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text='Back-calculate Cd')
        self.fields = 0
        self.mass_g_entry_2 = self.create_input_field(tab3, "Mass (g):", 1.0)
        self.fps_1_entry = self.create_input_field(tab3, "First Velocity (ft/s):", 300)
        self.dist_1_entry = self.create_input_field(tab3, "First Range (m):", 0)
        self.fps_2_entry = self.create_input_field(tab3, "Second Velocity (ft/s):", 240)
        self.dist_2_entry = self.create_input_field(tab3, "Second Range (m):", 10)
        self.cd_output = self.create_input_field(tab3, "Cd:", '')
        tk.Button(tab3, text="Calculate Cd", command=self.calculate_cd).grid(row=self.fields, column=0, columnspan=2)
        self.fields += 1

        tk.Label(tab3, text="Select a CSV file:").grid(row=self.fields, column=0, sticky='ew', columnspan=2, pady=(10, 0))
        self.fields += 1

        self.csv_files = ttk.Combobox(tab3, values=self.get_csv_files)
        self.csv_files.grid(row=self.fields, column=0, columnspan=2)
        self.csv_files.bind("<Button-1>", self.update_combobox)
        self.fields += 1

        tk.Button(tab3, text="Calculate All", command=self.calculate_cd_csv).grid(row=self.fields, column=0, columnspan=2)
        self.fields += 1

        tab3.rowconfigure(self.fields + 1, weight=1)
        tab3.columnconfigure(1, weight=1)

    def create_input_field(self, parent, label_text, default_value):
        """
        Create a labeled input field.

        :param parent: Parent widget
        :param label_text: Text for the label
        :param default_value: Default value for the input field
        :return: Entry widget
        """
        tk.Label(parent, text=label_text).grid(row=self.fields, column=0, sticky='e')
        entry = tk.Entry(parent)
        entry.grid(row=self.fields, column=1, sticky='ew')
        entry.insert(0, str(default_value))
        self.fields += 1
        return entry

    def run_simulation(self):
        """
        Run the simulation based on user input.
        """
        try:
            v = float(self.velocity_fps_entry.get()) * 0.3048
            x = float(self.velocity_distance_m_entry.get())
            mass = float(self.mass_g_entry.get()) / 1000
            Cd = float(self.drag_coefficient_entry.get())
            zero_distance = float(self.zero_distance_entry.get())

            ke = self.ke_entry.get()
            if ke != '' and float(ke) > 0:
                ke = float(self.ke_entry.get())
                v = np.sqrt(2 * ke / mass)
            else:
                ke = 0

            simulation = Sim(v=v, x=x, mass=mass, Cd=Cd, zero_range=zero_distance)
            simulation.run()
            sim_metadata = {
                "sim_id": self.sim_id,
                "velocity": v,
                "mass": mass,
                "Cd": Cd,
                "zero_distance": zero_distance,
                "ke": ke,
                "enabled": tk.BooleanVar(value=True),
                "measured_at": x,
                "label_text": ""
            }
            self.sim_id += 1

            messagebox.showinfo("Simulation Complete", "The simulation has been completed successfully.")

            enable_checkbox = tk.Checkbutton(self.sim_select_frame, text="", variable=sim_metadata["enabled"], command=self.update_plot, bg=self.colour_table[(sim_metadata["sim_id"] - 1) % len(self.colour_table)])
            enable_checkbox.grid(row=len(self.simulations), column=0, sticky='w')

            if ke > 0:
                sim_metadata["label_text"] = f"Simulation {sim_metadata['sim_id']}: {sim_metadata['ke']} J @ {sim_metadata['measured_at']} m, m={sim_metadata['mass']*1000} g, Cd={sim_metadata['Cd']:0.3f}"
                enable_checkbox.config(text=sim_metadata["label_text"])
            else:
                sim_metadata["label_text"] = f"Simulation {sim_metadata['sim_id']}: {sim_metadata['velocity']/0.3048} ft/s @ {sim_metadata['measured_at']} m, m={sim_metadata['mass']*1000} g, Cd={sim_metadata['Cd']:0.3f}"
                enable_checkbox.config(text=sim_metadata["label_text"])

            delete_button = tk.Button(self.sim_select_frame, text="Delete", command=lambda: self.delete_simulation(sim_metadata["sim_id"]))
            delete_button.grid(row=len(self.simulations), column=1, padx=5, pady=2)

            self.simulations.append((simulation, sim_metadata, enable_checkbox, delete_button))

            self.update_plot()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def calculate_cd(self):
        v1 = float(self.fps_1_entry.get()) * 0.3048
        x1 = float(self.dist_1_entry.get())
        v2 = float(self.fps_2_entry.get()) * 0.3048
        x2 = float(self.dist_2_entry.get())
        m = float(self.mass_g_entry_2.get()) / 1000
        Cd = self.reverse_sim(v1, x1, v2, x2, m)

        self.cd_output.delete(0, tk.END)  # Clear the current value
        self.cd_output.insert(0, f'{Cd:0.3f}')  # Insert the new value

    def reverse_sim(self, v1, x1, v2, x2, m):
        Cd_calc = 0.5
        scale_factor = 100
        mult = 5
        for _ in range(100):
            a = Sim(v=v1, Cd=Cd_calc, mass=m, x=x1)
            a.run()
            v_range = np.interp(x2, a.x, a.v)
            scale_factor = 1 - mult * (v2 - v_range) / v2
            print(f'{m}, {v1}, {v2} ,{v_range}, {scale_factor}, {Cd_calc}')
            if abs(1 - scale_factor) < 0.001:
                print('Converged!')
                break
            else:
                Cd_calc = Cd_calc * scale_factor
            mult = max(mult * 0.85, 1)
        return Cd_calc

    def calculate_cd_csv(self):
        try:
            written = 0
            fname = self.csv_files.get().removesuffix('.csv')
            with open(fname + '.csv', 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                header = header[0:5]
                header.append('Cd')
                with open(fname + '_proc.csv', 'w', newline='') as out:
                    writer = csv.writer(out)
                    writer.writerow(header)
                    for row in reader:
                        m = row[0]
                        v1 = row[1]
                        x1 = row[2]
                        v2 = row[3]
                        x2 = row[4]

                        Cd = self.reverse_sim(float(v1) * 0.3048, float(x1), float(v2) * 0.3048, float(x2), float(m) / 1000)

                        writer.writerow([m, v1, x1, v2, x2, Cd])
                        written += 1

            if written > 1:
                messagebox.showinfo("Reverse Simulations Complete", "The reverse simulations have been completed successfully.")
            elif written == 1:
                messagebox.showinfo("Reverse Simulation Complete", "The reverse simulation has been completed successfully.")
            elif written == 0:
                messagebox.showerror("Invalid Input", "No data was written to the CSV file.")

        except Exception as e:
            messagebox.showerror("Invalid Input", f'Something went wrong. {str(e)}')

    def update_combobox(self, _):
        """Update the values in the combobox when it is clicked."""
        self.csv_files['values'] = self.get_csv_files()

    def get_csv_files(self):
        """Get a list of all CSV files in the current working directory."""
        csv_files = [file for file in os.listdir(os.getcwd()) if file.endswith('.csv')]
        return csv_files

    def save_results(self):
        """
        Save the results of the simulations to files.
        """
        if self.simulations:
            for i, (sim, _, _, _) in enumerate(self.simulations):
                filename = f"simulation_results_{i + 1}.txt"
                sim.save_to_file(filename)
            messagebox.showinfo("Save Complete", "The simulation results have been saved.")
        else:
            messagebox.showwarning("No Simulation", "Please run the simulation first.")

    def create_plot_area(self, parent):
        """
        Create the plot area in the GUI.

        :param parent: Parent widget
        """
        plot_frame = ttk.LabelFrame(parent, text="Plot Area")
        plot_frame.grid(row=0, column=1, rowspan=6, padx=10, pady=10, sticky='nsew')

        # Configure the grid to make the plot area expand
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Make the canvas expand and fill the available space
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

    def delete_simulation(self, index):
        """
        Delete a simulation from the list.

        :param index: Index of the simulation to delete
        """

        sim_id = [i for i, (_, metadata, _, _) in enumerate(self.simulations) if metadata["sim_id"] == index]

        # Remove the simulation and its associated metadata and widgets
        _, _, enable_checkbox, delete_button = self.simulations.pop(sim_id[0])

        # Destroy the Tkinter widgets
        enable_checkbox.destroy()
        delete_button.destroy()

        # Reorganize the remaining widgets in the grid
        for i in range(len(self.simulations)):
            _, _, checkbox, button = self.simulations[i]
            checkbox.grid(row=i, column=0, sticky='w')
            button.grid(row=i, column=1, padx=5, pady=2)

        # Update the plot and simulation choices after deletion
        self.update_plot()

    def create_plot_options(self, parent):
        """
        Create the plot options area in the GUI.

        :param parent: Parent widget
        """
        plot_options_frame = ttk.LabelFrame(parent, text="Plot Options")
        plot_options_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.plot_type = tk.StringVar(value="velocity")
        tk.Radiobutton(plot_options_frame, text="Velocity", variable=self.plot_type, value="velocity", command=self.update_plot).pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(plot_options_frame, text="Kinetic Energy", variable=self.plot_type, value="ke", command=self.update_plot).pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(plot_options_frame, text="Y Position", variable=self.plot_type, value="y", command=self.update_plot).pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(plot_options_frame, text="Time", variable=self.plot_type, value="time", command=self.update_plot).pack(anchor='w', padx=5, pady=2)

        self.scale_var = tk.BooleanVar()
        self.scale_var.trace_add('write', self.update_scale_combobox)
        tk.Checkbutton(plot_options_frame, text="Scale as % of Sim:", variable=self.scale_var, command=self.update_plot).pack(anchor='w', padx=5, pady=2)

        self.scale_sim = ttk.Combobox(plot_options_frame, values=self.get_sim_names(), state='disabled')
        self.scale_sim.pack(anchor='w', padx=5, pady=2)
        self.scale_sim.bind("<Button-1>", self.update_scale_combobox)
        self.scale_sim.bind("<<ComboboxSelected>>", self.update_plot)

        self.marker_var = tk.BooleanVar()
        tk.Checkbutton(plot_options_frame, text="Set Point Markers", variable=self.marker_var, command=self.update_plot).pack(anchor='w', padx=5, pady=2)

        self.y_height_var = tk.BooleanVar(value=True)
        tk.Checkbutton(plot_options_frame, text="Limit Y Plot Height", variable=self.y_height_var, command=self.update_plot).pack(anchor='w', padx=5, pady=2)

    def update_scale_combobox(self, *args, **kwargs):
        self.scale_sim.config(state='normal' if self.scale_var.get() else 'disabled')
        old_value = self.scale_sim.get()
        self.scale_sim['values'] = self.get_sim_names()
        if old_value in self.scale_sim['values']:
            self.scale_sim.set(old_value)
        elif self.scale_sim['values'] != '':
            self.scale_sim.set(self.scale_sim['values'][0])
        else:
            self.scale_sim.set('')

    def get_sim_names(self):
        return [f'Simulation {metadata["sim_id"]}' for (_, metadata, _, _) in self.simulations if metadata["enabled"].get()]

    def update_plot(self, *args, **kwargs):
        """
        Update the plot based on the selected simulations and plot options.
        """
        self.update_scale_combobox()
        self.ax.clear()
        selected_sims = [i for i, (_, metadata, _, _) in enumerate(self.simulations) if metadata["enabled"].get()]

        if not selected_sims:
            self.canvas.draw()
            return

        base_sim = [sim for (sim, metadata, _, _) in self.simulations if metadata["sim_id"] == int(self.scale_sim.get().removeprefix('Simulation '))][0] if self.scale_var.get() else None

        for i in selected_sims:
            (sim, metadata, _, _) = self.simulations[i]
            if self.plot_type.get() == "velocity":
                y_data = sim.v
                if self.scale_var.get() and base_sim:
                    y_data = 100 * y_data / np.interp(sim.x, base_sim.x, base_sim.v)
                    y_label = "Velocity (%)"
                else:
                    y_data = y_data / 0.3048
                    y_label = "Velocity (ft/s)"
            elif self.plot_type.get() == "ke":
                y_data = sim.ke
                if self.scale_var.get() and base_sim:
                    y_data = 100 * y_data / np.interp(sim.x, base_sim.x, base_sim.ke)
                    y_label = "Kinetic Energy (%)"
                else:
                    y_label = "Kinetic Energy (J)"
            elif self.plot_type.get() == "y":
                y_data = sim.y_flat
                if self.scale_var.get() and base_sim:
                    # Note: we need to scale against the base simulation's y_flat, not the base simulation's y due to the zero crossings when fired upwards
                    y_data = 100 * (y_data - y_data[0]) / (np.interp(sim.x, base_sim.x, base_sim.y_flat) - base_sim.y_flat[0])
                    y_label = "Y Position (%)"
                else:
                    y_label = "Y Position (m)"
            elif self.plot_type.get() == "time":
                y_data = sim.time
                if self.scale_var.get() and base_sim:
                    y_data = 100 * y_data / np.interp(sim.x, base_sim.x, base_sim.time)
                    y_label = "Time (%)"
                else:
                    y_label = "Time (T)"

            self.ax.plot(sim.x, y_data, label=f"{self.simulations[i][1]['label_text']}", color=self.colour_table[(metadata["sim_id"] - 1) % len(self.colour_table)])
            if self.marker_var.get():
                self.ax.plot(metadata["measured_at"], np.interp(metadata["measured_at"], sim.x, y_data), color=self.colour_table[(metadata["sim_id"] - 1) % len(self.colour_table)], marker='o', markersize=5)

            if self.plot_type.get() == "y" and self.y_height_var.get() and not self.scale_var.get():
                self.ax.set_ylim(0, 2)

        self.ax.set_title(f"{y_label} vs. Distance")
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel(y_label)
        self.ax.legend()
        self.canvas.draw()

        # Refresh cursor
        if self.cursor:
            self.cursor.remove()
        self.cursor = mplcursors.cursor(self.ax, hover=True)


def main():
    root = tk.Tk()
    _ = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
