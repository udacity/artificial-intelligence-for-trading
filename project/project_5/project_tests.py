import numpy as np
import pandas as pd

from collections import OrderedDict

from tests import assert_output, project_test, assert_structure


@project_test
def test_get_documents(fn):
    # Test 1
    doc = '\nThis is inside the document\n' \
          'This is the text that should be copied'
    text = 'This is before the test document<DOCUMENT>{}</DOCUMENT>\n' \
           'This is after the document\n' \
           'This shouldn\t be included.'.format(doc)

    fn_inputs = {
        'text': text}
    fn_correct_outputs = OrderedDict([
        (
            'extracted_docs', [doc])])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)

    # Test 2
    ten_k_real_compressed_doc = '\n' \
        '<TYPE>10-K\n' \
        '<SEQUENCE>1\n' \
        '<FILENAME>test-20171231x10k.htm\n' \
        '<DESCRIPTION>10-K\n' \
        '<TEXT>\n' \
        '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">\n' \
        '<html>\n' \
        '	<head>\n' \
        '		<title>Document</title>\n' \
        '	</head>\n' \
        '	<body style="font-family:Times New Roman;font-size:10pt;">\n' \
        '...\n' \
        '<td><strong> Data Type:</strong></td>\n' \
        '<td>xbrli:sharesItemType</td>\n' \
        '</tr>\n' \
        '<tr>\n' \
        '<td><strong> Balance Type:</strong></td>\n' \
        '<td>na</td>\n' \
        '</tr>\n' \
        '<tr>\n' \
        '<td><strong> Period Type:</strong></td>\n' \
        '<td>duration</td>\n' \
        '</tr>\n' \
        '</table></div>\n' \
        '</div></td></tr>\n' \
        '</table>\n' \
        '</div>\n' \
        '</body>\n' \
        '</html>\n' \
        '</TEXT>\n'
    excel_real_compressed_doc = '\n' \
        '<TYPE>EXCEL\n' \
        '<SEQUENCE>106\n' \
        '<FILENAME>Financial_Report.xlsx\n' \
        '<DESCRIPTION>IDEA: XBRL DOCUMENT\n' \
        '<TEXT>\n' \
        'begin 644 Financial_Report.xlsx\n' \
        'M4$L#!!0    ( %"E04P?(\\#P    !,"   +    7W)E;,O+G)E;.MDD^+\n' \
        'MPD ,Q;]*F?L:5\#8CUYZ6U9_ )Q)OU#.Y,A$[%^>X>];+=44/ 87O+>CT?V\n' \
        '...\n' \
        'M,C,Q7V1E9BYX;6Q02P$"% ,4    " !0I4%,>V7[]F0L 0!(@A  %0\n' \
        'M        @ %N9@, 86UZ;BTR,#$W,3(S,5]L86(N>&UL4$L! A0#%     @\n' \
        'M4*5!3*U*Q:W#O0  U=\) !4              ( !!9,$ &%M>FXM,C Q-S$R\n' \
        '@,S%?<)E+GAM;%!+!08     !@ & (H!  #[4 4    !\n' \
        '\n' \
        'end\n' \
        '</TEXT>\n'
    real_compressed_text = '<SEC-DOCUMENT>0002014754-18-050402.txt : 20180202\n' \
        '<SEC-HEADER>00002014754-18-050402.hdr.sgml : 20180202\n' \
        '<ACCEPTANCE-DATETIME>20180201204115\n' \
        'ACCESSION NUMBER:		0002014754-18-050402\n' \
        'CONFORMED SUBMISSION TYPE:	10-K\n' \
        'PUBLIC DOCUMENT COUNT:		110\n' \
        'CONFORMED PERIOD OF REPORT:	20171231\n' \
        'FILED AS OF DATE:		20180202\n' \
        'DATE AS OF CHANGE:		20180201\n' \
        '\n' \
        'FILER:\n' \
        '\n' \
        '	COMPANY DATA:	\n' \
        '		COMPANY CONFORMED NAME:			TEST\n' \
        '		CENTRAL INDEX KEY:			0001018724\n' \
        '		STANDARD INDUSTRIAL CLASSIFICATION:	RANDOM [2357234]\n' \
        '		IRS NUMBER:				91236464620\n' \
        '		STATE OF INCORPORATION:			DE\n' \
        '		FISCAL YEAR END:			1231\n' \
        '\n' \
        '	FILING VALUES:\n' \
        '		FORM TYPE:		10-K\n' \
        '		SEC ACT:		1934 Act\n' \
        '		SEC FILE NUMBER:	000-2225413\n' \
        '		FILM NUMBER:		13822526583969\n' \
        '\n' \
        '	BUSINESS ADDRESS:	\n' \
        '		STREET 1:		422320 PLACE AVENUE\n' \
        '		CITY:			SEATTLE\n' \
        '		STATE:			WA\n' \
        '		ZIP:			234234\n' \
        '		BUSINESS PHONE:		306234534246600\n' \
        '\n' \
        '	MAIL ADDRESS:	\n' \
        '		STREET 1:		422320 PLACE AVENUE\n' \
        '		CITY:			SEATTLE\n' \
        '		STATE:			WA\n' \
        '		ZIP:			234234\n' \
        '</SEC-HEADER>\n' \
        '<DOCUMENT>{}</DOCUMENT>\n' \
        '<DOCUMENT>{}</DOCUMENT>\n' \
        '</SEC-DOCUMENT>\n'.format(ten_k_real_compressed_doc, excel_real_compressed_doc)

    fn_inputs = {
        'text': real_compressed_text}
    fn_correct_outputs = OrderedDict([
        (
            'extracted_docs', [ten_k_real_compressed_doc, excel_real_compressed_doc])])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_get_document_type(fn):
    doc = '\n' \
        '<TYPE>10-K\n' \
        '<SEQUENCE>1\n' \
        '<FILENAME>test-20171231x10k.htm\n' \
        '<DESCRIPTION>10-K\n' \
        '<TEXT>\n' \
        '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">\n' \
        '...'

    fn_inputs = {
        'doc': doc}
    fn_correct_outputs = OrderedDict([
        (
            'doc_type', '10-k')])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_lemmatize_words(fn):
    fn_inputs = {
        'words': ['cow', 'running', 'jeep', 'swimmers', 'tackle', 'throw', 'driven']}
    fn_correct_outputs = OrderedDict([
        (
            'lemmatized_words', ['cow', 'run', 'jeep', 'swimmers', 'tackle', 'throw', 'drive'])])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_get_bag_of_words(fn):
    def sort_ndarray(array):
        hashes = [hash(str(x)) for x in array]
        sotred_indicies = sorted(range(len(hashes)), key=lambda k: hashes[k])

        return array[sotred_indicies]

    fn_inputs = {
        'sentiment_words': pd.Series(['one', 'last', 'second']),
        'docs': [
            'this is a document',
            'this document is the second document',
            'last one']}
    fn_correct_outputs = OrderedDict([
        (
            'bag_of_words', np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1]]))])

    fn_out = fn(**fn_inputs)
    assert_structure(fn_out, fn_correct_outputs['bag_of_words'], 'bag_of_words')
    assert np.array_equal(sort_ndarray(fn_out.T), sort_ndarray(fn_correct_outputs['bag_of_words'].T)), \
        'Wrong value for bag_of_words.\n' \
        'INPUT docs:\n{}\n\n' \
        'OUTPUT bag_of_words:\n{}\n\n' \
        'A POSSIBLE CORRECT OUTPUT FOR bag_of_words:\n{}\n'\
        .format(fn_inputs['docs'], fn_out, fn_correct_outputs['bag_of_words'])


@project_test
def test_get_jaccard_similarity(fn):
    fn_inputs = {
        'bag_of_words_matrix': np.array([
                [0, 1, 1, 0, 0, 0, 1],
                [0, 1, 2, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0]])}
    fn_correct_outputs = OrderedDict([
        (
            'jaccard_similarities', [0.7142857142857143, 0.0])])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_get_tfidf(fn):
    def sort_ndarray(array):
        hashes = [hash(str(x)) for x in array]
        sotred_indicies = sorted(range(len(hashes)), key=lambda k: hashes[k])

        return array[sotred_indicies]

    fn_inputs = {
        'sentiment_words': pd.Series(['one', 'last', 'second']),
        'docs': [
            'this is a document',
            'this document is the second document',
            'last one']}
    fn_correct_outputs = OrderedDict([
        (
            'tfidf', np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.70710678, 0.70710678]]))])

    fn_out = fn(**fn_inputs)
    assert_structure(fn_out, fn_correct_outputs['tfidf'], 'tfidf')
    assert np.isclose(sort_ndarray(fn_out.T), sort_ndarray(fn_correct_outputs['tfidf'].T)).all(), \
        'Wrong value for tfidf.\n' \
        'INPUT docs:\n{}\n\n' \
        'OUTPUT tfidf:\n{}\n\n' \
        'A POSSIBLE CORRECT OUTPUT FOR tfidf:\n{}\n'\
        .format(fn_inputs['docs'], fn_out, fn_correct_outputs['tfidf'])


@project_test
def test_get_cosine_similarity(fn):
    fn_inputs = {
        'tfidf_matrix': np.array([
                [0.0,           0.57735027, 0.57735027, 0.0,        0.0,        0.0,        0.57735027],
                [0.0,           0.32516555, 0.6503311,  0.0,        0.42755362, 0.42755362, 0.32516555],
                [0.70710678,    0.0,        0.0,        0.70710678, 0.0,        0.0,        0.0]])}
    fn_correct_outputs = OrderedDict([
        (
            'cosine_similarities', [0.75093766927060945, 0.0])])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)
