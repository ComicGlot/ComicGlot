import os
corpus_file = "E:/ko.txt"

def split_file(input_file, output_dir, num_parts):
    """
    Split a large file into a specified number of parts.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory where the split files will be saved.
        num_parts (int): Number of parts to split the file into.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)

    lines_per_file = total_lines // num_parts
    if total_lines % num_parts != 0:
        lines_per_file += 1  # Ensure all lines are included

    print(f"Total lines: {total_lines}")
    print(f"Lines per file: {lines_per_file}")

    with open(input_file, 'r', encoding='utf-8') as file:
        for part in range(num_parts):
            part_file_path = os.path.join(output_dir, f"part_{part}.txt")
            with open(part_file_path, 'w', encoding='utf-8') as out_file:
                for _ in range(lines_per_file):
                    line = file.readline()
                    if not line:
                        break
                    out_file.write(line)
            print(f"Saved: {part_file_path}")

split_file(corpus_file, "./splitted_corpus", num_parts=10)