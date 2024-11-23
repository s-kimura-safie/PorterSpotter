/*
 * (c) 2021 Safie Inc.
 *
 * NOTICE: No part of this file may be reproduced, stored
 * in a retrieval system, or transmitted, in any form, or by any means,
 * electronic, mechanical, photocopying, recording, or otherwise,
 * without the prior consent of Safie Inc.
 */

#include "LinearSumAssignment.hpp"
#include <cfloat> // dblmax
#include <cmath>  // fabs()

LinearSumAssignment::LinearSumAssignment(const cv::Mat m) : mat(m), coveredCols(m.cols, false), coveredRows(m.rows, false)
{
    nrows = m.rows;
    ncols = m.cols;

    starMat = cv::Mat::zeros(nrows, ncols, CV_8U);
    primeMat = cv::Mat::zeros(nrows, ncols, CV_8U);
    if (nrows <= ncols)
    {
        isRowBigThanCol = false;
        minDim = nrows;
    }
    else
    {
        isRowBigThanCol = true;
        minDim = ncols;
    }
}

/// @brief 線形割当 (Hungarian Algorithm) の実行
/// @retval 線形割当のマッチング結果。行列の番号が格納されたリスト。
void LinearSumAssignment::ComputeAssociation(std::list<RowCol> &association)
{
    step1();
    // bool is_finish = false;
    bool isFirst = true;
    bool skip235 = false;
    while (1)
    {
        if (!skip235)
        {
            step2(isFirst);
            isFirst = false;
            if (step3())
            {
                break;
            };
        }
        skip235 = step4();
        if (!skip235)
        {
            step5();
        }
        else
        {
            step6();
        }
    }
    putStMat2Vec(association);
}

/// @brief 行の最小値を探し出し、各要素から引く
void LinearSumAssignment::step1()
{
    if (isRowBigThanCol)
    {
        mat = mat.t();
    }
    double minVal;
    // step1: 行の中から最小値を探し、その値を行の要素から引く
    for (int r = 0; r < mat.rows; r++)
    {
        // 最小値の探し出し
        minVal = mat.at<double>(r, 0);
        for (int c = 0; c < mat.cols; c++)
        {
            double v = mat.at<double>(r, c);
            if (v < minVal)
            {
                minVal = v;
            }
        }
        // 最小値を引く
        for (int c = 0; c < mat.cols; c++)
        {
            mat.at<double>(r, c) = mat.at<double>(r, c) - minVal;
        }
    }
    if (isRowBigThanCol)
    {
        mat = mat.t();
    }
}

/// @brief Star Zeroを作成する
/// @param isFirst 1回目かどうかのフラグ
void LinearSumAssignment::step2(const bool isFirst)
{
    // step2a: 各列の最初の0を Star Zero として定義する
    for (int r = 0; r < nrows; r++)
    {
        for (int c = 0; c < ncols; c++)
        {
            if (!isFirst)
            {
                if (starMat.at<unsigned char>(r, c))
                {
                    coveredCols[c] = true;
                    break;
                }
            }
            else
            {
                if (fabs(mat.at<double>(r, c)) < DBL_EPSILON)
                {
                    if (!coveredCols[c])
                    {
                        starMat.at<unsigned char>(r, c) = 1;
                        coveredCols[c] = true;
                        if (isRowBigThanCol)
                        {
                            coveredRows[r] = true;
                        }
                        break;
                    }
                }
            }
        }
    }
    if (isRowBigThanCol)
    {
        // 行のフラグは false に
        for (int r = 0; r < nrows; r++)
        {
            coveredRows[r] = false;
        }
    }
}

/// @brief すべての列の covered flag が立てられていれば終了。そうでなければ次のステップへ。
bool LinearSumAssignment::step3() const
{
    int cnt = 0;
    for (int c = 0; c < ncols; c++)
    {
        if (coveredCols[c])
        {
            cnt++;
        }
    }
    if (cnt == minDim) // アルゴリズム終了
    {
        return true;
    }
    else
    {
        return false;
    }
}

/// @brief 行でも列でもcoverされていない0を見つけ、Prime Zeroとする。
/// このPrime Zeroを含む列にStar Zeroがなかった場合はstep5へ。
/// そうでなければ行をcoverし列をuncoverする。
bool LinearSumAssignment::step4()
{
    bool zerosFlg = false;
    for (int c = 0; c < ncols; c++)
    {
        if (coveredCols[c])
        {
            continue;
        }
        for (int r = 0; r < nrows; r++)
        {
            if ((!coveredRows[r]) && (fabs(mat.at<double>(r, c)) < DBL_EPSILON))
            {
                // 行、列ともにCoverされていない0に対しPrime Zeroの作成
                primeMat.at<unsigned char>(r, c) = 1;

                int starCol;
                // 現在の列でStar Zeroを探す
                starCol = findInMat(starMat, r, false);

                // Star Zeroが列になければ
                if (starCol == -1)
                {
                    currRow = r;
                    currCol = c;
                    return false;
                }
                else
                {
                    coveredRows[r] = true;
                    coveredCols[starCol] = false;
                    zerosFlg = true;
                    break;
                }
            }
        }
    }
    if (zerosFlg)
    {
        return step4();
    }
    else
    {
        return true;
    }
}

/// @brief step4で見つけたPrime Zeroに対し同じ列のStar Zeroを探し出す。
/// 見つけたStar Zeroと同じ行のPrime Zeroを探し出す。
/// 同じ列にStar ZeroがないPrime Zeroが見つかるまで繰り返す。
/// Star Zero のStarを外し、Prime ZeroをStar Zeroとする。
/// 全Prime Zeroをなくし、行のCoveredフラグをなくす。
void LinearSumAssignment::step5()
{
    cv::Mat starMatTmp;
    starMatTmp = cv::Mat::zeros(nrows, ncols, CV_8U);
    // Star Zeroを更新するためのテンポラリ行列の作成
    for (int r = 0; r < nrows; r++)
    {
        for (int c = 0; c < ncols; c++)
        {
            starMatTmp.at<unsigned char>(r, c) = starMat.at<unsigned char>(r, c);
        }
    }
    // 現在のPrime Zeroを次期Star Zeroに
    starMatTmp.at<unsigned char>(currRow, currCol) = 1;
    // 同列のStarZeroを探す
    int starRow;
    int starCol = currCol;
    starRow = findInMat(starMat, starCol, true);
    while (starRow != -1)
    {
        // Star Zeroのフラグを外す
        starMatTmp.at<unsigned char>(starRow, starCol) = 0;

        // 同行のPrime Zeroを探す
        int primeCol;
        int primeRow = starRow;
        primeCol = findInMat(primeMat, primeRow, false);
        // Prime ZeroをStar Zeroにする
        starMatTmp.at<unsigned char>(primeRow, primeCol) = 1;

        // 同列のStar Zeroを探す
        starCol = primeCol;
        starRow = findInMat(starMat, starCol, true);
    }
    // Prime Zeroを0に
    // Star Zeroを更新
    // 行のCoverをなくす
    for (int r = 0; r < nrows; r++)
    {
        for (int c = 0; c < ncols; c++)
        {
            primeMat.at<unsigned char>(r, c) = 0;
            starMat.at<unsigned char>(r, c) = starMatTmp.at<unsigned char>(r, c);
        }
        coveredRows[r] = false;
    }
    return;
}

/// @brief 行列ともにCoverされていない要素の最小値を探しだし
/// Coverされている列の値には足す。
/// Coverされていない行の値からは引く。
void LinearSumAssignment::step6()
{
    double minVal, v;

    // 行列ともにCoverされていない要素の最小値を探す
    minVal = DBL_MAX;
    for (int r = 0; r < nrows; r++)
    {
        if (coveredRows[r])
        {
            continue;
        }
        for (int c = 0; c < ncols; c++)
        {
            if (!coveredCols[c])
            {
                v = mat.at<double>(r, c);
                if (v < minVal)
                {
                    minVal = v;
                }
            }
        }
    }

    // Coverされている行の値には足す
    // Coverされていない列の値からは引く
    for (int r = 0; r < nrows; r++)
    {
        for (int c = 0; c < ncols; c++)
        {
            if (coveredRows[r])
            {
                mat.at<double>(r, c) += minVal;
            }
            if (!coveredCols[c])
            {
                mat.at<double>(r, c) -= minVal;
            }
        }
    }

    return;
}

/// @brief 行列から値を探す関数
/// @param m 探す対象の行列
/// @param c 探す行または列
/// @param isRow 行を探すかどうかのフラグ。falseだったら列を探す。
/// @retval 見つけた行または列
int LinearSumAssignment::findInMat(const cv::Mat m, const int c, const bool isRow)
{
    int n = -1;
    int size;
    if (isRow)
    {
        size = m.rows;
        for (int i = 0; i < size; i++)
        {
            if (m.at<unsigned char>(i, c))
            {
                n = i;
            }
        }
    }
    else
    {
        size = m.cols;
        for (int i = 0; i < size; i++)
        {
            if (m.at<unsigned char>(c, i))
            {
                n = i;
            }
        }
    }
    return n;
}

/// @brief starMat行列の値をリストに格納して返す
/// @param[out] res 結果を格納するRowCol構造体型のリスト
void LinearSumAssignment::putStMat2Vec(std::list<RowCol> &res) const
{
    for (int r = 0; r < nrows; r++)
    {
        for (int c = 0; c < ncols; c++)
        {
            if (starMat.at<unsigned char>(r, c))
            {
                RowCol tmpRes;
                tmpRes.row = r;
                tmpRes.col = c;

                res.push_back(tmpRes);
                break;
            }
        }
    }
}
