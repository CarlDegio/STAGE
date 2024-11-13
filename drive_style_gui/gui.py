import PySimpleGUI as sg


class StyleGUI:
    def __init__(self, range = (0, 1)):
        initial_value = (range[0] + range[1]) / 2
        layout = [
            [sg.Column([
                [sg.Text("driving style value setting", justification='center', pad=(0, 10))],
                [sg.Slider(range=range, resolution=0.01, orientation='h',
                           size=(20, 15), key='-SLIDER-', default_value=initial_value)]
            ], element_justification='center', justification='center')]
        ]
        self.window = sg.Window('driving style', layout, resizable=True)

    def read_style(self):
        event, values = self.window.read(timeout=10)  # 每隔100ms读取窗口事件和拖动条值
        return values['-SLIDER-']


if __name__ == '__main__':
    gui = StyleGUI()
    for i in range(100):
        print(gui.read_style())
