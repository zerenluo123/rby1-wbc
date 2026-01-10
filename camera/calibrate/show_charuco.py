import argparse
import cv2
import numpy as np

def make_board(squares_x, squares_y, square_len_px, marker_len_ratio, dict_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_len_px = max(1, int(round(square_len_px * marker_len_ratio)))

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=1.0,      # "unit" length; we'll treat 1.0 as one square
        markerLength=marker_len_ratio,  # relative to squareLength
        dictionary=aruco_dict
    )

    # Render at desired pixel size
    w = squares_x * square_len_px
    h = squares_y * square_len_px
    img = board.generateImage((w, h), marginSize=max(1, square_len_px // 10), borderBits=1)
    return img

def put_overlay(img_bgr, text_lines):
    y = 30
    for line in text_lines:
        cv2.putText(img_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        y += 28

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares_x", type=int, default=7)
    parser.add_argument("--squares_y", type=int, default=5)
    parser.add_argument("--dict", default="DICT_5X5_1000")
    parser.add_argument("--marker_ratio", type=float, default=0.75, help="markerLength / squareLength (0.6~0.8 typical)")
    parser.add_argument("--start_square_px", type=int, default=120, help="initial pixels per square on screen")
    parser.add_argument("--window", default="charuco")
    parser.add_argument("--save_png", default="", help="optional path to save current board png (press 's' too)")
    args = parser.parse_args()

    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    }
    if args.dict not in dict_map:
        raise ValueError(f"Unknown dict {args.dict}. Choose from: {list(dict_map.keys())}")

    square_px = args.start_square_px

    # Fullscreen window
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        board_img = make_board(
            args.squares_x, args.squares_y, square_px, args.marker_ratio, dict_map[args.dict]
        )

        # Convert to BGR for overlay text
        vis = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

        overlay = [
            f"ChArUco {args.squares_x}x{args.squares_y}  {args.dict}  marker_ratio={args.marker_ratio:.2f}",
            "Controls: [+]/[-] scale squares, [s] save PNG, [q]/[esc] quit",
            "Tip: measure one displayed square with a ruler; use that as square_length in calibration (mm->m).",
            f"Current: {square_px} px per square (screen-dependent)"
        ]
        # put_overlay(vis, overlay)

        cv2.imshow(args.window, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key in (ord('+'), ord('=')):
            square_px = min(600, square_px + 10)
        elif key in (ord('-'), ord('_')):
            square_px = max(30, square_px - 10)
        elif key == ord('s'):
            out = args.save_png if args.save_png else "charuco_board.png"
            cv2.imwrite(out, board_img)
            print(f"Saved: {out}")
        # Any other key: just redraw

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
