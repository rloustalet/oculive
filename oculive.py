import os
import numpy as np
import cv2
from astropy.io import fits
from skimage.feature import peak_local_max
from PyQt6 import QtWidgets, QtGui, QtCore


class ImageViewer(QtWidgets.QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.pixmap_item = None
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._zoom = 0
        self._empty = True

    def has_image(self):
        return not self._empty

    def fitInView(self, scale=True):
        if self.pixmap_item is None:
            return
        rect = QtCore.QRectF(self.pixmap_item.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_image():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                self._zoom = 0

    def set_image(self, qimg):
        self._zoom = 0
        pixmap = QtGui.QPixmap.fromImage(qimg)
        if self.pixmap_item:
            self.pixmap_item.setPixmap(pixmap)
        else:
            self.pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self.pixmap_item)
        self._empty = False
        self.fitInView()

    def clear_image(self):
        if self.pixmap_item:
            self._scene.removeItem(self.pixmap_item)
            self.pixmap_item = None
            self._empty = True

    def wheelEvent(self, event):
        if not self.has_image():
            return
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 20:
            self._zoom = 20
            return

        self.scale(zoom_factor, zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.has_image():
            self.fitInView()


class LiveStacker(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.folder = None
        self.files = []
        self.index = 0
        self.ref_img = None  # Référence (np.ndarray) - can be grayscale or color
        self.stacked = None
        self.stack_count = 0
        self.stack_buffer = []
        self.processing = False
        self.black_level = 0
        self.white_level = 255
        self.stack_method = "median"
        self.auto_stretch_flag = False
        self.dark_img = None  # Image dark (np.ndarray)
        self.flat_img = None

        self.bayer_pattern = None  # Store detected Bayer pattern string

        self.init_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.load_new_images)
        self.timer.setInterval(1500)

    def init_ui(self):
        self.setWindowTitle("Oculive")
        self.resize(1100, 900)

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        toolbar = QtWidgets.QToolBar("Actions")
        self.addToolBar(toolbar)

        toolbar_bottom = QtWidgets.QToolBar("Stretch")
        self.addToolBar(QtCore.Qt.ToolBarArea.BottomToolBarArea, toolbar_bottom)

        btn_choose = QtGui.QAction("Select folder", self)
        btn_choose.triggered.connect(self.choose_folder)
        toolbar.addAction(btn_choose)


        self.btn_start = QtGui.QAction("Start", self)
        self.btn_start.triggered.connect(self.toggle_processing)
        toolbar.addAction(self.btn_start)

        self.btn_reset = QtGui.QAction("Reset", self)
        self.btn_reset.setEnabled(False)
        self.btn_reset.triggered.connect(self.reset_stack)
        toolbar.addAction(self.btn_reset)

        self.btn_save = QtGui.QAction("Save", self)
        self.btn_save.setEnabled(False)
        self.btn_save.triggered.connect(self.save_stack)
        toolbar.addAction(self.btn_save)

        toolbar.addSeparator()

        self.check_dark = QtWidgets.QCheckBox("Darks")
        self.check_dark.setChecked(False)
        self.check_dark.stateChanged.connect(self.choose_dark_file)
        self.check_dark.stateChanged.connect(lambda state: self.status.showMessage("Dark activé" if state else "Dark désactivé"))

        toolbar.addWidget(self.check_dark)

        self.check_flat = QtWidgets.QCheckBox("Flats")
        self.check_flat.setChecked(False)
        toolbar.addWidget(self.check_flat)
        self.check_flat.stateChanged.connect(self.choose_flat_file)
        self.check_flat.stateChanged.connect(lambda state: self.status.showMessage("Flat activé" if state else "Flat désactivé"))

        self.combo_bayer = QtWidgets.QComboBox()
        self.combo_bayer.addItems(["None", "RGGB", "BGGR", "GRBG", "GBRG"])
        self.combo_bayer.setCurrentText("RGGB")  # Default to RGGB
        self.combo_bayer.currentTextChanged.connect(lambda text: self.status.showMessage(f"Bayer Pattern sélectionné : {text}"))
        toolbar.addWidget(QtWidgets.QLabel("Bayer Pattern : "))
        toolbar.addWidget(self.combo_bayer)

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(["Mean", "Median"])
        self.combo_method.setCurrentText("Median") # Default to Median
        self.combo_method.currentTextChanged.connect(self.change_stack_method)
        toolbar.addWidget(QtWidgets.QLabel("Method : "))
        toolbar.addWidget(self.combo_method)

        self.check_auto_stretch = QtWidgets.QCheckBox("Auto Stretch")
        self.check_auto_stretch.setChecked(False)
        self.check_auto_stretch.stateChanged.connect(self.auto_stretch)
        toolbar.addWidget(self.check_auto_stretch)


        self.slider_black = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_black.setRange(0, 255)
        self.slider_black.setValue(self.black_level)
        self.slider_black.valueChanged.connect(self.update_contrast)
        toolbar_bottom.addWidget(QtWidgets.QLabel("Black Level"))
        toolbar_bottom.addWidget(self.slider_black)

        self.slider_white = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_white.setRange(0, 255)
        self.slider_white.setValue(self.white_level)
        self.slider_white.valueChanged.connect(self.update_contrast)
        toolbar_bottom.addWidget(QtWidgets.QLabel("White Level"))
        toolbar_bottom.addWidget(self.slider_white)

        self.status = self.statusBar()
        self.lbl_count = QtWidgets.QLabel("Stacked images: 0")
        self.lbl_current = QtWidgets.QLabel("File : None")
        self.status.addPermanentWidget(self.lbl_current)
        self.status.addPermanentWidget(self.lbl_count)


    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Fits folder", QtCore.QDir.homePath())
        if folder:
            self.folder = folder
            self.load_file_list()
            self.status.showMessage(f"Selected folder : {self.folder}")
            self.btn_save.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.viewer.clear_image()

    def auto_stretch(self):
        if self.check_auto_stretch.isChecked():
            self.auto_stretch_flag = True
            self.status.showMessage("Auto Stretch activated")
            self.slider_black.setEnabled(False)
            self.slider_white.setEnabled(False)
            if self.stacked is not None:
                self.status.showMessage("Updating contrast levels...")
                self.white_level = (np.mean(self.stacked) + 3 * np.std(self.stacked)).astype(np.uint8)
                self.black_level = np.min(self.stacked)
                self.slider_black.setValue(self.black_level)
                self.slider_white.setValue(self.white_level)
                # For simplicity, just reload the current displayed image with new levels:
                self.index -= 1
                self.stack_count -= 1
                self.stack_buffer.pop()  # Remove last image from buffer
                self.load_new_images()

        else:
            self.auto_stretch_flag = False
            self.status.showMessage("Auto Stretch deactivated")
            self.slider_black.setEnabled(True)
            self.slider_white.setEnabled(True)
    def choose_dark_file(self):
        if self.dark_img is None:
            # Choisir le fichier dark
            dark_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Dark file", "", "FITS Files (*.fit*)")
            if dark_file:
                with fits.open(dark_file) as hdul:
                    dark_data = hdul[0].data.astype(np.float32)[::-1, ::-1]  # Inverser l'image
                    # Normaliser le dark
                    dark_data = self.normalize_to_uint8(dark_data)
                    self.dark_img = dark_data
                self.status.showMessage(f"Loaded dark : {dark_file}")
                self.check_dark.setChecked(True)
            else:
                self.status.showMessage("No dark file selected")

    def choose_flat_file(self):
        if self.flat_img is None:
            # Choisir le fichier flat
            flat_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Flat file", "", "FITS Files (*.fit*)")
            if flat_file:
                with fits.open(flat_file) as hdul:
                    flat_data = hdul[0].data.astype(np.float32)[::-1, ::-1]  # Inverser l'image
                    # Normaliser le flat
                    flat_data = self.normalize_to_uint8(flat_data)
                    self.flat_img = flat_data
                self.status.showMessage(f"Loaded flat : {flat_file}")
                self.check_flat.setChecked(True)
            else:
                self.status.showMessage("No flat file selected")

    def load_file_list(self):
        # Liste uniquement les fichiers .fits dans le dossier
        self.files = sorted([f for f in os.listdir(self.folder) if f.lower().endswith('.fits' ) or f.lower().endswith('.fit')])
        self.index = 0

    def set_reference_image(self):
        if not self.files:
            self.status.showMessage("No FITS files found in the selected folder")
            return
        filename = os.path.join(self.folder, self.files[0])
        img, bayer = self.read_fits_image(filename)
        if img.ndim == 3:
            img = self.normalize_background_to_gray(img)
        self.ref_img = img
        self.ref_img = np.clip(self.ref_img, np.min(self.ref_img), np.max(self.ref_img))
        self.ref_img = (self.ref_img - np.min(self.ref_img)) / (np.max(self.ref_img) - np.min(self.ref_img)) * 255
        self.ref_img = np.uint8(self.ref_img)
        # Store the Bayer pattern if detected
        self.bayer_pattern = bayer
        self.status.showMessage(f"Reference image : {self.files[0]} (Bayer: {self.bayer_pattern})")
        self.stack_count = 0

    def toggle_processing(self):
        if self.processing:
            self.timer.stop()
            self.processing = False
            self.btn_start.setText("Start")
            self.status.showMessage("Stop")
            self.btn_save.setEnabled(True)
            self.btn_reset.setEnabled(True)
        else:
            self.set_reference_image()
            self.stack_count = 0
            self.timer.start()
            self.processing = True
            self.btn_start.setText("Stop")
            self.status.showMessage("Processing images...")
            self.btn_save.setEnabled(False)
            self.btn_reset.setEnabled(False)

    def load_new_images(self):
        current_files = sorted([f for f in os.listdir(self.folder) if f.lower().endswith('.fits') or f.lower().endswith('.fit')])
        if len(current_files) > len(self.files):
            # Mise à jour de la liste des fichiers si de nouveaux ont été ajoutés
            self.files = current_files
            self.status.showMessage(f"New files detected, total: {len(self.files)}")
        # Charge l'image suivante du dossier, empile et affiche
        if self.index >= len(self.files):
            self.status.showMessage("Waiting for new images...")
            return
        filename = os.path.join(self.folder, self.files[self.index])
        img, bayer = self.read_fits_image(filename)
        # Vérifier si Bayer correspond à celui de référence (sinon on ignore)
        if bayer != self.bayer_pattern:
            self.status.showMessage(f"Different Bayer pattern for {self.files[self.index]} : {bayer} != {self.bayer_pattern}, image ignored")
            self.index += 1
            return
        if img.shape != self.ref_img.shape:
            self.status.showMessage(f"Different size for {self.files[self.index]}, image ignored")
            self.index += 1
            return

        if self.stack_buffer is None:
            self.stack_buffer.append(img)
            self.stack_count = 1
        else:
            img_shifted = self.align_image_translation(self.ref_img, img)
            self.stack_buffer.append(img_shifted)
            self.stack_count += 1

        # Affichage
        if len(self.stack_buffer) > 0:
            if self.stack_method == "median" and len(self.stack_buffer) > 0:
                stacked = np.median(np.array(self.stack_buffer), axis=0)
            elif self.stack_method == "mean":
                stacked = np.mean(np.array(self.stack_buffer), axis=0)
        else:
            stacked = img

        if stacked.ndim == 3:
            # Correct background for RGB
            stacked = self.normalize_background_to_gray(stacked)
        
        if self.auto_stretch_flag:
            if self.stacked is not None:
                self.status.showMessage("Updating contrast levels...")
                self.white_level = (np.mean(self.stacked) + 3 * np.std(self.stacked)).astype(np.uint8)
                self.black_level = np.min(self.stacked)
                self.slider_black.setValue(self.black_level)
                self.slider_white.setValue(self.white_level)

        stacked = np.clip(stacked, self.black_level, self.white_level)
        stacked = (stacked - np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 255
        stacked = np.uint8(stacked)

        self.stacked = stacked

        self.viewer.set_image(self.get_qimage(stacked))

        self.lbl_count.setText(f"Stacked images: {self.stack_count}")
        self.lbl_current.setText(f"File: {self.files[self.index]}")

        self.index += 1

    def reset_stack(self):
        self.stack_buffer.clear()
        self.files = []
        self.stack_count = 0
        self.index = 0
        self.ref_img = None
        self.viewer.clear_image()
        self.lbl_count.setText("Stacked images: 0")
        self.status.showMessage("Stack reseted")
        self.btn_save.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.check_dark.setChecked(False)
        self.check_flat.setChecked(False)
        self.dark_img = None
        self.flat_img = None

    def save_stack(self):
        if self.stack_count == 0:
            self.status.showMessage("No images stacked yet")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save stacked image", "", "FITS Files (*.fits)")
        if fname:
            # Reconstituer l'image finale
            if len(self.stack_buffer) > 0:
                if self.stack_method == "median":
                    stacked = np.median(np.array(self.stack_buffer), axis=0)
                elif self.stack_method == "mean":
                    stacked = np.mean(np.array(self.stack_buffer), axis=0)
            else:
                stacked = self.ref_img

            hdu = fits.PrimaryHDU(stacked.astype(np.float32))
            # Ajouter header d'origine (optionnel)
            hdu.header['STACKED'] = (True, 'Image stacked with Oculive')
            if self.bayer_pattern:
                hdu.header['BAYERPAT'] = self.bayer_pattern
            hdu.writeto(fname, overwrite=True)
            self.status.showMessage(f"Stack saved as {fname}")

    def update_contrast(self):
        self.black_level = self.slider_black.value()
        self.white_level = self.slider_white.value()
        # Met à jour l'affichage selon les nouveaux niveaux
        if self.stack_count > 0:
            self.status.showMessage("Updating contrast levels...")
            # For simplicity, just reload the current displayed image with new levels:
            self.index -= 1
            self.stack_count -= 1
            self.stack_buffer.pop()  # Remove last image from buffer
            self.load_new_images()


    def compute_background_rgb_means(self, rgb_image, low_percentile=20):
        gray = np.mean(rgb_image, axis=2)
        mask = gray < np.percentile(gray, low_percentile)
        mean_r = np.mean(rgb_image[:, :, 0][mask])
        mean_g = np.mean(rgb_image[:, :, 1][mask])
        mean_b = np.mean(rgb_image[:, :, 2][mask])
        return mean_r, mean_g, mean_b

    def normalize_background_to_gray(self, rgb_image):
        mean_r, mean_g, mean_b = self.compute_background_rgb_means(rgb_image)
        avg = (mean_r + mean_g + mean_b) / 3.0
        gain_r = avg / mean_r
        gain_g = avg / mean_g
        gain_b = avg / mean_b
        corrected = np.zeros_like(rgb_image)
        corrected[:, :, 0] = np.clip(rgb_image[:, :, 0] * gain_r, 0, 255)
        corrected[:, :, 1] = np.clip(rgb_image[:, :, 1] * gain_g, 0, 255)
        corrected[:, :, 2] = np.clip(rgb_image[:, :, 2] * gain_b, 0, 255)
        return corrected.astype(np.uint8)
    
    def find_shift_with_phase_correlation(self, ref_img_gray, img_gray):
        # OpenCV phase correlation pour estimer un déplacement translationnel précis
        shift, response = cv2.phaseCorrelate(np.float32(ref_img_gray), np.float32(img_gray))
        dx, dy = shift
        return dx, dy

    def align_image_translation(self, ref_img, img_to_align):
        # Convertir en niveaux de gris
        if ref_img.ndim == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = ref_img
            img_gray = img_to_align

        dx, dy = self.find_shift_with_phase_correlation(ref_gray, img_gray)
        # Matrice de translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(img_to_align, M, (ref_img.shape[1], ref_img.shape[0]))
        return aligned
    

    def change_stack_method(self, text):
        self.stack_method = text.lower()
        self.status.showMessage(f"Stack method : {text}")
        

    def read_fits_image(self, filename):
        """
        Lit une image FITS, 
        applique le dématriçage si nécessaire.
        Retourne l'image finale en RGB (np.ndarray uint8) ou en grayscale.
        """
        with fits.open(filename) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(np.uint16)[::-1, ::-1]

            #application des darks et flats si activés
            if self.check_dark.isChecked() and self.dark_img is not None:
                # soustraction du dark
                data = data.astype(np.float32) - self.dark_img.astype(np.float32)
            if self.check_flat.isChecked() and self.flat_img is not None and self.dark_img is not None:
                # division par le flat
                data = data.astype(np.float32) / ((self.flat_img.astype(np.float32) - self.dark_img.astype(np.float32)) / np.median(self.flat_img))
            # Normalisation des données
            data = np.clip(data, 0, 65535)  # Clip to valid range for uint16

            # Si l'image est en 16 bits, on la convertit en 8 bits

            if data.dtype == np.uint16:
                # Normalisation entre 0-255
                data = self.normalize_to_uint8(data)
            elif data.dtype != np.uint8:
                # Si ce n'est pas du uint8, on normalise
                data = self.normalize_to_uint8(data)
       

            # utilisation du Bayer pattern
            bayer_pattern = self.combo_bayer.currentText()
            if bayer_pattern == "None":
                bayer_pattern = None

            # S'il n'y a pas de pattern, on retourne directement l'image en grayscale (uint8)
            if bayer_pattern is None:
                # On normalise et on convertit en uint8
                norm_img = self.normalize_to_uint8(data)
                return norm_img, None

            # Map Bayer pattern string FITS -> OpenCV demosaicing code
            # Patterns usually are 'RGGB', 'BGGR', 'GRBG', 'GBRG'
            pattern_map = {
                'RGGB': cv2.COLOR_BayerRG2RGB,
                'BGGR': cv2.COLOR_BayerBG2RGB,
                'GRBG': cv2.COLOR_BayerGR2RGB,
                'GBRG': cv2.COLOR_BayerGB2RGB,
            }
            if bayer_pattern not in pattern_map:
                # Pattern non reconnu
                norm_img = self.normalize_to_uint8(data)
                return norm_img, None

            # Dématriçage OpenCV
            # OpenCV attend un np.uint8 ou uint16, on adapte les données
            # Si data est uint16, on peut garder, OpenCV gère
            img_bayer = data
            # Conversion OpenCV
            img_rgb = cv2.cvtColor(img_bayer, pattern_map[bayer_pattern])

            # Normaliser entre 0-255 (convertir en uint8)
            img_rgb8 = self.normalize_to_uint8(img_rgb)
            
            # Appliquer les niveaux de noir et blanc
            img_rgb8 = np.clip(img_rgb8, np.min(img_rgb8), np.max(img_rgb8))
            img_rgb8 = (img_rgb8 - np.min(img_rgb8)) / (np.max(img_rgb8) - np.min(img_rgb8)) * 255
            img_rgb8 = np.uint8(img_rgb8)
            return img_rgb8, bayer_pattern

    def normalize_to_uint8(self, img):
        """
        Normalise une image numpy (quelque soit le type) entre 0-255 uint8
        """
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            return np.zeros(img.shape, dtype=np.uint8)
        img_norm = (img - min_val) / (max_val - min_val) * 255
        return img_norm.astype(np.uint8)

    def get_qimage(self, img):
        """
        Convertit np.ndarray en QImage.
        Supporte images grayscale ou RGB.
        """
        if img.ndim == 2:
            # Grayscale
            qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format.Format_Grayscale8)
            return qimg.copy()
        elif img.ndim == 3:
            # RGB, OpenCV uses BGR, we have converted to RGB so direct copy works
            height, width, channels = img.shape
            if channels == 3:
                qimg = QtGui.QImage(img.data, width, height, img.strides[0], QtGui.QImage.Format.Format_RGB888)
                return qimg.copy()
            else:
                # Unexpected channels count
                return None
        else:
            return None


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('logo_oculive.ico'))
    window = LiveStacker()
    window.show()
    sys.exit(app.exec())
