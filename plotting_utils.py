import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    # Исправление для новых версий matplotlib
    try:
        # Современный способ для новых версий matplotlib
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        # Преобразуем RGBA в RGB
        data = data[:, :, :3]
    except AttributeError:
        try:
            # Для промежуточных версий matplotlib
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Fallback для старых версий
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    try:
        # Проверяем входные данные
        if alignment is None or alignment.size == 0:
            print("⚠️ Alignment пустой, создаем заглушку")
            alignment = np.zeros((10, 10))
        
        # Убеждаемся, что alignment 2D
        if len(alignment.shape) != 2:
            print(f"⚠️ Неожиданная размерность alignment: {alignment.shape}")
            alignment = alignment.squeeze()
            if len(alignment.shape) != 2:
                alignment = np.zeros((10, 10))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(alignment, aspect='auto', origin='lower',
                       interpolation='none', cmap='Blues')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        if info is not None:
            xlabel += '\n\n' + info
        plt.xlabel(xlabel)
        plt.ylabel('Encoder timestep')
        plt.title(f'Alignment Matrix ({alignment.shape[0]}x{alignment.shape[1]})')
        plt.tight_layout()

        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data
    except Exception as e:
        print(f"❌ Ошибка в plot_alignment_to_numpy: {e}")
        # Создаем заглушку
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'Alignment Error:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        data = save_figure_to_numpy(fig)
        plt.close()
        return data


def plot_spectrogram_to_numpy(spectrogram):
    try:
        # Проверяем входные данные
        if spectrogram is None or spectrogram.size == 0:
            print("⚠️ Spectrogram пустой, создаем заглушку")
            spectrogram = np.zeros((80, 100))
        
        # Убеждаемся, что spectrogram 2D
        if len(spectrogram.shape) != 2:
            print(f"⚠️ Неожиданная размерность spectrogram: {spectrogram.shape}")
            spectrogram = spectrogram.squeeze()
            if len(spectrogram.shape) != 2:
                spectrogram = np.zeros((80, 100))
        
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none', cmap='viridis')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Mel Channels")
        plt.title(f'Mel Spectrogram ({spectrogram.shape[0]}x{spectrogram.shape[1]})')
        plt.tight_layout()

        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data
    except Exception as e:
        print(f"❌ Ошибка в plot_spectrogram_to_numpy: {e}")
        # Создаем заглушку
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, f'Spectrogram Error:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        data = save_figure_to_numpy(fig)
        plt.close()
        return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    try:
        # Проверяем входные данные
        if gate_targets is None or gate_outputs is None:
            print("⚠️ Gate данные пустые, создаем заглушку")
            gate_targets = np.zeros(100)
            gate_outputs = np.random.random(100)
        
        # Убеждаемся, что данные 1D
        if len(gate_targets.shape) > 1:
            gate_targets = gate_targets.flatten()
        if len(gate_outputs.shape) > 1:
            gate_outputs = gate_outputs.flatten()
        
        # Обрезаем до одинаковой длины
        min_len = min(len(gate_targets), len(gate_outputs), 1000)  # Максимум 1000 точек
        gate_targets = gate_targets[:min_len]
        gate_outputs = gate_outputs[:min_len]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.7,
                   color='green', marker='+', s=2, label='target')
        ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.7,
                   color='red', marker='.', s=2, label='predicted')
        
        plt.xlabel("Frames")
        plt.ylabel("Gate State")
        plt.title(f'Gate Outputs (Green=target, Red=predicted) - {len(gate_targets)} frames')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()

        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data
    except Exception as e:
        print(f"❌ Ошибка в plot_gate_outputs_to_numpy: {e}")
        # Создаем заглушку
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, f'Gate Error:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        data = save_figure_to_numpy(fig)
        plt.close()
        return data
