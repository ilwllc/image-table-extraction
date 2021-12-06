import cv2
import os
import sys
import csv

import pytesseract

def find_tables(image):
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = image
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    
    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    SCALE = 5
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    mask = horizontally_dilated + vertically_dilated
    ret, contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    MIN_TABLE_AREA = 1e5
    contours = [c for c in contours if cv2.contourArea(c) > MIN_TABLE_AREA]
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.1 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]

    images = [image[y:y+h, x:x+w] for x, y, w, h in bounding_rects]
    return images

def extract_cell_images_from_table(image):
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    
    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    vertical = horizontal = img_bin.copy()
    SCALE = 20
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    mask = horizontally_dilated + vertically_dilated
    ret, contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.05 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    
    # Filter out contours that aren't rectangular. Those that aren't rectangular
    # are probably noise.
    approx_rects = [p for p in approx_polys if len(p) == 4]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]
    
    # Filter out rectangles that are too narrow or too short.
    MIN_RECT_WIDTH = 40
    MIN_RECT_HEIGHT = 10
    bounding_rects = [
        r for r in bounding_rects if MIN_RECT_WIDTH < r[2] and MIN_RECT_HEIGHT < r[3]
    ]
    
    # The largest bounding rectangle is assumed to be the entire table.
    # Remove it from the list. We don't want to accidentally try to OCR
    # the entire table.
    largest_rect = max(bounding_rects, key=lambda r: r[2] * r[3])
    bounding_rects = [b for b in bounding_rects if b is not largest_rect]
    
    cells = [c for c in bounding_rects]
    def cell_in_same_row(c1, c2):
        c1_center = c1[1] + c1[3] - c1[3] / 2
        c2_bottom = c2[1] + c2[3]
        c2_top = c2[1]
        return c2_top < c1_center < c2_bottom
    
    orig_cells = [c for c in cells]
    rows = []
    while cells:
        first = cells[0]
        rest = cells[1:]
        cells_in_same_row = sorted(
            [
                c for c in rest
                if cell_in_same_row(c, first)
            ],
            key=lambda c: c[0]
        )
    
        row_cells = sorted([first] + cells_in_same_row, key=lambda c: c[0])
        rows.append(row_cells)
        cells = [
            c for c in rest
            if not cell_in_same_row(c, first)
        ]
    
    # Sort rows by average height of their center.
    def avg_height_of_center(row):
        centers = [y + h - h / 2 for x, y, w, h in row]
        return sum(centers) / len(centers)
    
    rows.sort(key=avg_height_of_center)
    cell_images_rows = []
    for row in rows:
        cell_images_row = []
        for x, y, w, h in row:
            cell_images_row.append(image[y:y+h, x:x+w])
        cell_images_rows.append(cell_images_row)
    return cell_images_rows

def extract_cells(f, table):
    print("Type of file input for extract_cells is: ", type(f), "Type of table input is: ", type(table))
    results = []
    directory, filename = os.path.split(f)
    rows = extract_cell_images_from_table(table)
    cell_img_dir = os.path.join(directory, "cells")
    os.makedirs(cell_img_dir, exist_ok=True)
    paths = []
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_filename = "{:03d}-{:03d}.png".format(i, j)
            path = os.path.join(cell_img_dir, cell_filename)
            cv2.imwrite(path, cell)
            paths.append(path)
    return paths

def ocr_image(image_folder, tess_args):
    """
    OCR the image and output the text to a file with an extension that is ready
    to be used in Tesseract training (.gt.txt).
    Returns the name of the text file that contains the text.
    """
    directory, foldername = os.path.split(image_folder)
    ocr_data_dir = os.path.join(directory, "ocr_data")
    os.makedirs(ocr_data_dir, exist_ok=True)
    paths = []
    for file in sorted(os.listdir(image_folder)):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".png"):
            filename_sans_ext, ext = os.path.splitext(filename)
            file_string = image_folder + "/" + filename
            image = cv2.imread(file_string, cv2.IMREAD_GRAYSCALE)
            # os.remove(file_string)
            out_txtpath = os.path.join(ocr_data_dir, "{}.txt".format(filename_sans_ext))
            # cv2.imwrite(out_imagepath, cropped)
            configs=("".join(tess_args))
            txt = pytesseract.image_to_string(image, config=configs)
            with open(out_txtpath, "w") as txt_file:
                txt_file.write(txt.encode("ascii", "ignore").decode("ascii"))
            paths.append(out_txtpath)
        else:
            continue
    return paths

def text_files_to_list(files):
    """Files must be sorted lexicographically
    Filenames must be <row>-<colum>.txt.
    000-000.txt
    000-001.txt
    001-000.txt
    etc...
    """
    rows = []
    punc = '''!()[]{};:'"\,<>.@^&*_~'''
    for f in files:
        directory, filename = os.path.split(f)
        with open(f) as of:
            txt = of.read()
            for symbol in punc:
                if symbol in txt:
                    txt = txt.replace(symbol, "")
            txt_stripped = txt.replace("\n", " ")
        row, column = map(int, filename.split(".")[0].split("-"))
        if row == len(rows):
            rows.append([])
        rows[row].append(txt_stripped)
        # os.remove(f)

    return rows

def cell_shift(list_of_rows):

    for i in range(len(list_of_rows)-1):
        if len(list_of_rows[i+1]) < max(len(x) for x in list_of_rows):
            for j in range(len(list_of_rows[i])):
                if "  " in list_of_rows[i][j]:
                    a = list_of_rows[i][j].split("  ", 1)[0]
                    print(a)
                    b = list_of_rows[i][j].split("  ", 1)[1]
                    print(b)
                    list_of_rows[i][j] = a
                    list_of_rows[i+1].insert(j, b)
    
    return list_of_rows

def main(f):
    #Import image of the table
    directory, image_file = os.path.split(f)
    filename = os.path.splitext(image_file)[0]
    #Convert image to proper format
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    print("Image imported: ", type(image))
    # Find tables in image
    tables = find_tables(image)
    table = tables[0]
    print("The type of the table extracted is: ", type(table))
    #Extract cells from image
    cells = extract_cells(f, table)
    print(cells)
    cell_img_dir = os.path.join(directory, "cells")
    # OCR's data into folder \ocr-data
    texts = ocr_image(cell_img_dir, "")

    print(texts)
    # Compile output from ocr_image into a list of lists
    list_of_txt = text_files_to_list(texts)

    # Shift any misaligned cells back into appropriate rows
    #final_list = cell_shift(list_of_txt)

    # Create output dir
    output_dir = os.path.join(directory, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    # Write output to .csv file
    with open(os.path.join(output_dir, filename+".csv"), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(list_of_txt)
        csv_file.close()

if __name__ == "__main__":
    print(sys.argv[1])
    main(sys.argv[1])
