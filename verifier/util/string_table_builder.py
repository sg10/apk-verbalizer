class StringTable:

    def __init__(self, separator=" | ", align="left"):
        self._rows = {}
        self._i_row = 0
        self._i_col = 0
        self._headline = {}
        self._align = align
        self._headline_separator = "-"

        self._global_max_cell_length = 25
        self._max_cell_length = {}
        self._separator = separator

    def add_cell(self, value, i_row=None, i_col=None):
        i_row = self._i_row if i_row is None else i_row
        i_col = self._i_col if i_col is None else i_col

        self._rows[i_row] = self._rows.get(i_row, {})
        self._rows[i_row][i_col] = value

        self._i_col = i_col + 1

    def add_cells(self, values):
        for value in values:
            self.add_cell(value)

    def new_row(self):
        if self._i_col != 0:
            self._i_col = 0
            self._i_row = self._i_row + 1

    def set_headline(self, headline, i_col=None):
        if type(headline) is list:
            self._headline = {i: h for i, h in enumerate(headline)}
        elif type(headline) is dict:
            self._headline = headline
        else:
            if len(self._headline) == 0:
                i_col = 0
            elif i_col is None:
                i_col = max(self._headline.keys()) + 1
            else:
                i_col = i_col
            self._headline[i_col] = headline

    def set_cell_length(self, i_cell, length):
        if i_cell >= 0:
            self._max_cell_length[i_cell] = length
        elif i_cell < 0 and len(self.cols()) > 0:
            i_cell = self.cols()[i_cell]
            self._max_cell_length[i_cell] = length

    def create_table(self, return_rows=False):
        if len(self._rows) == 0:
            return "" if not return_rows else []

        self.new_row()

        headline_and_rows = [self._headline] + list(self._rows.values())

        column_lengths = {}
        for row in headline_and_rows:
            for i_cell, value in row.items():
                value = self.format_cell(value)
                length = self._max_cell_length.get(i_cell, self._global_max_cell_length)
                length = min(len(value), length)
                length = max(length, column_lengths.get(i_cell, 0))
                column_lengths[i_cell] = length

        string_table = []
        min_col, max_col = min(column_lengths.keys()), max(column_lengths.keys())
        for row in headline_and_rows:
            string_row = []
            for i_cell in range(min_col, max_col+1):
                col_length = column_lengths.get(i_cell, -1)
                if col_length == -1:
                    continue
                value = self.format_cell(row.get(i_cell, ""), col_length)
                string_row.append(value)
            string_table.append(string_row)

        rows_as_strings = [self._separator.join(string_row) for string_row in string_table]

        if self._headline_separator:
            line = self._headline_separator*len(rows_as_strings[0])
            rows_as_strings.insert(1, line)

        if return_rows:
            return rows_as_strings
        else:
            return "\n".join(rows_as_strings)

    def format_cell(self, value, max_length=None):
        if type(value) is float:
            value = "%.2f" % value
        elif type(value) is int:
            value = "%d" % value
        elif type(value) is str:
            if max_length is not None:
                value = value[:max_length]
        else:
            value = str(value)
            if max_length is not None:
                value = value[:max_length]

        if max_length is not None:
            if len(value) > max_length:  # integer/float string too long
                value = "<?>"

            if self._align == 'center':
                value = value.center(max_length)
            elif self._align == 'right':
                value = value.rjust(max_length)
            elif self._align == 'left':
                value = value.ljust(max_length)

        return value

    def cols(self):
        cols = []
        for row_cols in self._rows.values():
            cols = cols + list(row_cols.keys())
        return list(set(cols))

    def __str__(self):
        return self.create_table()


if __name__ == "__main__":
    table = StringTable()

    print(table.create_table())

    table.add_cell("aaaa")
    table.add_cell("b")
    table.add_cell("dd")
    table.new_row()
    table.add_cell("a")
    table.add_cell("cc")
    table.add_cell("cc")
    table.add_cell("cc")

    table.add_cell("new!!!!!", i_row=0, i_col=2)
    table.add_cell("new!!!!!", i_row=5, i_col=2)
    table.add_cell(798023489072389748723498., i_row=5, i_col=2)
    table.add_cell(33123, i_row=3, i_col=4)
    table.add_cell("new!!!!!", i_row=5, i_col=7)

    table.set_headline(["H1", "H2123123123123123123123123"])

    print(table.cols())

    print(table.create_table())
