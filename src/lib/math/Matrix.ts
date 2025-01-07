type Generator = (i: number, j: number) => number;

export default class Matrix<Row extends number = number, Col extends number = number> {

    data: number[][] = [];

    constructor(public rows: Row, public cols: Col, data: number[][] | Generator = []) {
        this.rows = rows;
        this.cols = cols;
        if (typeof data === 'function') {
            this.data = Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => data(i, j)));
        } else if (data.length === 0) {
            this.data = Array.from({ length: rows }, () => Array(cols).fill(0).map(() => Math.random()));
        } else {
            this.data = data;
        }
    }

    static multiply<A extends number, B extends number, C extends number>(a: Matrix<A, B>, b: Matrix<B, C>): Matrix<A, C> {
        const bTranspose = b.transpose();
        return new Matrix(a.rows, b.cols, a.data.map(row => bTranspose.data.map(col => Matrix.dot(row, col))));
    }
    multiply(b: Matrix) {
        return Matrix.multiply(this, b);
    }

    static divide<A extends number, B extends number>(a: Matrix<A, B>, b: Matrix<A, B>) {
        return a.map((v, i, j) => v / b.get(i, j));
    }
    divide(b: Matrix<Row, Col>){
        return Matrix.divide(this, b);
    }

    static dot(a: number[], b: number[]): number {
        return a.reduce((sum, cell, index) => sum + cell * b[index], 0);
    }

    static softmax<Row extends number, Col extends number>(matrix: Matrix<Row, Col>): Matrix<Row, Col> {
        return new Matrix<Row, Col>(matrix.rows, matrix.cols, matrix.data.map(row => {
            const max = Math.max(...row);
            const exps = row.map(value => Math.exp(value - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(value => value / sum);
        }));
    }

    static add<Row extends number, Col extends number>(a: Matrix<Row, Col>, b: Matrix<Row, Col>): Matrix<Row, Col> {
        return new Matrix(a.rows, a.cols, a.data.map((row, i) => row.map((cell, j) => cell + b.data[i][j])));
    }
    add(b: Matrix<Row, Col>) {
        return Matrix.add(this, b);
    }

    static subtract<Row extends number, Col extends number>(a: Matrix<Row, Col>, b: Matrix<Row, Col>): Matrix<Row, Col> {
        return new Matrix(a.rows, a.cols, a.data.map((row, i) => row.map((cell, j) => cell - b.data[i][j])));
    }
    subtract(b: Matrix<Row, Col>) {
        return Matrix.subtract(this, b);
    }

    static multiplyScalar<Row extends number, Col extends number>(a: Matrix<Row, Col>, scalar: number): Matrix<Row, Col> {
        return new Matrix(a.rows, a.cols, a.data.map(row => row.map(cell => cell * scalar)));
    }
    multiplyScalar(scalar: number) {
        return Matrix.multiplyScalar(this, scalar);
    }

    transpose(): Matrix<Col, Row> {
        return new Matrix(this.cols, this.rows, this.data[0].map((_, colIndex) => this.data.map(row => row[colIndex])));
    }

    map(mapFunc: (value: number, i: number, j: number) => number) {
        return new Matrix(this.rows, this.cols, this.data.map((row, i) => row.map((val, j) => mapFunc(val, i, j))));
    }

    get(i: number, j: number) {
        return this.data[i][j];
    }
    
    mean(axis: "col"): Matrix<1, Col>;
    mean(axis: "row"): Matrix<Row, 1>;
    mean(): number;
    mean(axis?: "row" | "col"): number | Matrix<Row, 1> | Matrix<1, Col> {
        if (axis === "row") {
            return new Matrix(this.rows, 1, this.data.map(row => [row.reduce((sum, value) => sum + value, 0) / this.cols]));
        } else if (axis === "col") {
            return new Matrix(1, this.cols, [this.data[0].map((_, colIdx) => 
                this.data.reduce((sum, row) => sum + row[colIdx], 0) / this.rows
            )]);
        } else {
            const totalSum = this.data.reduce((sum, row) => sum + row.reduce((innerSum, value) => innerSum + value, 0), 0);
            const totalCount = this.rows * this.cols;
            return totalSum / totalCount;
        }
    }

    variance(axis: "col"): Matrix<1, Col>;
    variance(axis: "row"): Matrix<Row, 1>;
    variance(): number;
    variance(axis?: "row" | "col"): number | Matrix<Row, 1> | Matrix<1, Col> {
        if (axis === "row") {
            return new Matrix(this.rows, 1, this.data.map(row => {
                const mean = row.reduce((sum, value) => sum + value, 0) / this.cols;
                return [row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / this.cols];
            }));
        } else if (axis === "col") {
            return new Matrix(1, this.cols, [this.data[0].map((_, colIdx) => {
                const mean = this.data.reduce((sum, row) => sum + row[colIdx], 0) / this.rows;
                return this.data.reduce((sum, row) => sum + (row[colIdx] - mean) ** 2, 0) / this.rows;
            })]);
        } else {
            const mean = this.mean() as number;
            const totalSum = this.data.reduce((sum, row) => sum + row.reduce((innerSum, value) => innerSum + (value - mean) ** 2, 0), 0);
            const totalCount = this.rows * this.cols;
            return totalSum / totalCount;
        }
    }

    reduce(reduceFunc: (accumulator: number, currentValue: number, i: number, j: number) => number, initialValue: number = 0): number {
        return this.data.reduce((accumulator, row, i) => 
            row.reduce((innerAccumulator, value, j) => reduceFunc(innerAccumulator, value, i, j), accumulator)
        , initialValue);
    }

    sum(axis: "row"): Matrix<Row, 1>;
    sum(axis: "col"): Matrix<1, Col>;
    sum(): number;
    sum(axis?: "row" | "col"): number | Matrix<Row, 1> | Matrix<1, Col> {
        if (axis === "row") {
            return new Matrix(this.rows, 1, this.data.map(row => [row.reduce((sum, value) => sum + value, 0)]));
        } else if (axis === "col") {
            return new Matrix(1, this.cols, [this.data[0].map((_, colIdx) => 
                this.data.reduce((sum, row) => sum + row[colIdx], 0)
            )]);
        } else {
            return this.data.reduce((sum, row) => sum + row.reduce((innerSum, value) => innerSum + value, 0), 0);
        }
    }

    row(i: number) {
        return this.data[i];
    }
    col(j: number) {
        return this.data.map(row => row[j]);
    }

}