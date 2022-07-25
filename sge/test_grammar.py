import unittest
import warnings




class Test(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=DeprecationWarning)

    def test_read_file(self):
        import sge.grammar
        g = sge.grammar.Grammar()
        g.set_path('grammars/regression.txt')
        g.read_grammar()
        print(str(g))
        #print(g.mapping(genome, mapping_numbers, needs_python_filter=True))
        #output = """<start> ::= <expr>\n<expr> ::= <expr><op><expr> | (<expr><op><expr>) | <pre_op>(<expr>) | <var>\n<op> ::= + | - | * | \eb_div_\eb\n<pre_op> ::= sin | cos | _exp_ | _log_\n<var> ::= x[0] | 1.0"""

        output = """<start> ::= <expr>
<expr> ::= <expr><op><expr> | (<expr><op><expr>) | <pre_op>(<expr>) | <var>
<op> ::= + | - | * | \eb_div_\eb
<pre_op> ::= sin | cos | _exp_ | _log_
<var> ::= x[0] | 1.0
"""

        print(output)



        self.assertEqual(str(g), output, "Error: Grammars Differ")

    def test_simple_mapping(self):
        import sge.grammar
        genome = [[0], [0, 3, 3], [0], [], [1, 1]]
        mapping_numbers = [0] * len(genome)
        g = sge.grammar.Grammar()
        g.set_path('grammars/regression.txt')
        g.read_grammar()
        g.get_non_terminals()
        g.count_number_of_options_in_production()
        g.compute_non_recursive_options()
        g.mapping(mapping_numbers)
        #print(str(g))
        self.assertEqual(g.mapping(genome, mapping_numbers, needs_python_filter=True), ('1.0+1.0', 4), "Error")

#    def test_longer_genomes(self):
        import sge.grammar
        genome = [[0], [0, 2, 1, 0, 2, 3, 0, 1, 0, 0, 2, 0, 2, 0, 2, 0, 3, 1, 0, 3, 0, 3, 3, 1, 2, 3, 2, 3, 0, 0, 1, 2, 1, 3, 3, 2, 1, 3, 3, 0, 0, 1, 3, 3, 1, 3, 3, 2, 1, 3, 3, 2, 1, 0, 1, 3, 3, 2, 3, 1, 1, 3, 3, 3, 2, 3, 2, 1, 0, 3, 1, 0, 3, 1, 0, 3, 1, 2, 3, 2, 3, 2, 2, 0, 3, 3, 0, 3, 0, 2, 0, 3, 3, 0, 1, 1, 3, 3, 1, 3, 3, 0, 3, 2, 3, 3, 1, 0, 1, 0, 1, 3, 1, 1, 3, 2, 1, 3, 2, 3, 0, 3, 2, 0, 0, 3, 3, 2, 3, 1, 2, 1, 3, 3, 0, 1, 3, 1, 0, 3, 3, 3, 3, 0, 2, 2, 0, 2, 1, 3, 1, 3, 3, 2, 0, 0, 3, 3, 0, 3, 3, 0, 0, 3, 2, 3, 3, 2, 3, 3, 0, 3, 3, 3, 0, 1, 3, 2, 3, 0, 3, 1, 0, 1, 2, 2, 1, 2, 2, 3, 2, 1, 0, 2, 1, 3, 1, 3, 3, 2, 2, 2, 3, 2, 1, 0, 1, 3, 3, 1, 3, 3, 0, 1, 3, 3, 1, 3, 3, 1, 0, 3, 0, 3, 2, 2, 3, 2, 2, 2, 1, 0, 2, 2, 1, 3, 3, 0, 2, 0, 3, 3, 2, 3, 0, 1, 3, 1, 2, 3, 1, 3, 3, 0, 2, 1, 3, 3, 1, 0, 3, 3, 0, 3, 3, 2, 3, 0, 0, 2, 0, 2, 0, 2, 0, 3, 0, 1, 2, 3, 2, 3, 2, 3, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 3, 3, 0, 1, 2, 0, 3, 3, 2, 0, 3, 1, 3, 1, 3, 2, 3, 1, 2, 2, 1, 0, 0, 3, 3, 2, 3, 0, 0, 3, 3, 2, 3, 2, 1, 0, 0, 3, 1, 3, 3, 3, 2, 0, 2, 3, 1, 3, 3, 2, 0, 0, 0, 1, 2, 0, 2, 1, 3, 3, 1, 2, 3, 0, 3, 3, 2, 2, 0, 0, 3, 3, 2, 3, 0, 3, 3, 2, 0, 1, 1, 2, 1, 3, 3, 1, 1, 3, 3, 1, 3, 3, 2, 1, 2, 3, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 1, 3, 1, 1, 2, 1, 0, 2, 2, 1, 1, 0, 2, 0, 3, 2, 1, 2, 3, 2, 3, 2, 2, 2, 1, 3, 2, 3, 1, 2, 3, 2, 3, 0, 2, 1, 2, 0, 3, 0, 2, 3, 3, 3, 1, 1, 0, 3, 0, 1, 1, 3, 3, 1, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 0, 1, 2, 0, 2, 2, 3, 1, 3, 0, 3, 3, 3, 3, 1, 2, 2, 3, 0, 3, 0, 0, 3, 1, 0, 3, 0, 3, 2, 3, 0, 1, 0, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 2, 3, 2, 3, 3, 3, 0, 0, 1, 0, 2, 0, 3, 0, 0, 3, 3, 0, 3, 3, 1, 3, 1, 2, 3, 2, 1, 3, 3, 2, 2, 2, 2, 1, 3, 3, 2, 0, 3, 0, 0, 2, 2, 3, 1, 2, 3, 2, 3, 0, 0, 3, 2, 3, 2, 0, 3, 3, 1, 0, 3, 0, 2, 2, 1, 1, 3, 3, 0, 3, 3, 1, 3, 1, 2, 2, 3, 1, 0, 3, 3, 1, 3, 3, 0, 0, 3, 2, 1, 0, 0, 3, 3, 3, 2, 1, 3, 3, 3, 3, 0, 0, 2, 2, 3, 3, 2, 2, 2, 0, 1, 1, 1, 0, 2, 1, 0, 3, 3, 1, 2, 3, 2, 1, 3, 3, 2, 0, 1, 1, 2, 3, 1, 3, 3, 1, 2, 3, 2, 3, 1, 2, 1, 3, 3, 2, 2, 3, 0, 3, 0, 0, 3, 3, 1, 3, 0, 3, 0, 1, 3, 3, 1, 3, 3, 3, 1, 1, 2, 3, 1, 2, 2, 3, 3, 0, 1, 3, 1, 0, 0, 2, 3, 3, 3, 3, 2, 0, 0, 1, 3, 2, 1, 3, 3, 3, 2, 0, 1, 2, 3, 3, 2, 2, 3, 1, 0, 3, 2, 2, 0, 2, 0, 3, 2, 0, 3, 3, 3, 2, 3], [0, 2, 0, 3, 3, 2, 0, 3, 0, 2, 3, 3, 2, 1, 1, 2, 0, 3, 2, 0, 1, 3, 2, 3, 3, 0, 0, 0, 3, 0, 2, 0, 1, 0, 2, 1, 0, 1, 3, 3, 0, 3, 2, 2, 3, 0, 2, 2, 0, 1, 0, 3, 2, 0, 2, 0, 0, 3, 2, 2, 0, 3, 0, 1, 3, 0, 3, 3, 0, 1, 0, 1, 3, 3, 2, 1, 1, 3, 2, 0, 2, 2, 0, 1, 2, 0, 0, 0, 3, 2, 1, 0, 1, 3, 1, 3, 0, 2, 1, 2, 0, 1, 1, 3, 3, 0, 0, 0, 3, 1, 0, 3, 1, 3, 3, 0, 3, 2, 2, 0, 2, 3, 1, 0, 2, 2, 3, 3, 0, 1, 3, 3, 2, 0, 3, 3, 0, 2, 2, 1, 3, 1, 2, 2, 3, 1, 2, 2, 3, 0, 0, 0, 1, 2, 0, 3, 3, 3, 2, 0, 1, 2, 1, 0, 1, 1, 0, 0, 1, 2, 1, 0, 2, 1, 1, 3, 1, 3, 1, 3, 3, 1, 2, 0, 3, 1, 1, 1, 2, 1, 0, 3, 3, 2, 0, 0, 0, 3, 1, 2, 2, 0, 2, 1, 0, 1, 1, 1, 1, 3, 0, 3, 2, 3, 3, 1, 1, 1, 3, 0, 1, 1, 1, 0, 1, 0, 3, 3, 2, 2, 0, 2, 0, 2, 2, 0, 3, 3, 3, 2, 3, 0, 2, 1, 2, 2, 1, 2, 3, 2, 0, 0, 3, 0, 1, 3, 0, 2, 3, 1, 1, 0, 1, 3, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0, 2, 3, 2, 2, 1, 0, 3], [0, 2, 0, 2, 3, 3, 0, 1, 0, 3, 3, 2, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 2, 3, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 3, 1, 0, 1, 3, 3, 3, 3, 1, 3, 2, 2, 2, 2, 0, 3, 2, 3, 2, 3, 2, 0, 1, 1, 3, 1, 1, 1, 1, 1, 3, 0, 1, 3, 1, 0, 1, 2, 2, 3, 1, 3, 1, 3, 3, 0, 1, 1, 3, 1, 2, 2, 3, 3, 2, 1, 1, 0, 3, 3, 3, 3, 3, 2, 2, 0, 1, 1, 2, 2, 3, 3, 2, 0, 3, 3, 2, 3, 2, 0, 0, 0, 1, 0, 0, 3, 3, 0, 1, 3, 2, 3, 1, 3, 2, 0, 0, 1, 0, 2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 0, 0, 2, 0, 3, 2, 1, 3, 3, 0, 0, 1, 0, 0, 0, 3, 1, 3, 1, 3, 3, 1], [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]]
        g = sge.grammar.set_path('grammars/regression.txt')
        mapping_numbers = [0] * len(genome)
        self.assertEqual(g.mapping(genome, mapping_numbers, needs_python_filter=True), ('sin((_exp_(1.0)+(sin(_exp_(_log_(1.0*(x[0]+1.0|_div_|1.0|_div_|(_log_(1.0)*sin(x[0]))))+(cos((x[0]|_div_|1.0))+sin((1.0*1.0)))|_div_|(x[0]|_div_|x[0])*(x[0]-1.0)-_log_((1.0*x[0]))+_log_(((1.0|_div_|x[0])*_exp_(1.0)+((1.0-x[0])|_div_|x[0]))))*sin(1.0))|_div_|sin((1.0|_div_|(1.0+(1.0+(cos(x[0])+_exp_(x[0]))|_div_|cos(sin(1.0+x[0])))*1.0+sin(1.0-1.0)+((1.0*x[0])-(x[0]+x[0]))-x[0]|_div_|sin(1.0))|_div_|1.0))+(((x[0]|_div_|((1.0*sin((1.0*sin(1.0))))|_div_|x[0]+_exp_(x[0]*x[0]*_log_(1.0))))+(cos((x[0]-1.0))+(1.0|_div_|(1.0*1.0+x[0]))*x[0])+cos(_exp_(cos((1.0+(x[0]|_div_|x[0])))*cos(1.0*1.0+1.0|_div_|1.0)))+1.0-sin(1.0)|_div_|1.0)+cos(1.0)|_div_|x[0])|_div_|1.0+x[0])-x[0]+(1.0-cos(1.0))|_div_|x[0]|_div_|((cos(_exp_((cos(sin(1.0))*_exp_((cos((1.0-(x[0]-x[0])))|_div_|_log_(cos(sin(x[0])))*cos(((1.0+1.0)*(1.0*x[0])+(x[0]-x[0])*(1.0+1.0))))))))+(1.0+x[0]|_div_|_log_(_log_(1.0))*_log_(_log_(cos((_log_(_exp_((x[0]-1.0)))+_exp_(1.0-x[0])|_div_|_exp_(1.0)-(1.0|_div_|(_exp_(x[0])+(x[0]*x[0])))-sin((x[0]*x[0]))+(1.0-1.0-1.0|_div_|1.0)))))))|_div_|_log_(x[0])+_exp_(_log_(_exp_(1.0+(_log_(1.0)+_exp_(1.0))|_div_|sin(1.0))-((cos(1.0)+1.0)|_div_|cos(_log_((1.0-(x[0]|_div_|x[0]))))))|_div_|(cos(x[0]+x[0])|_div_|cos(x[0]*(1.0*(x[0]+cos(x[0])))))*(cos(cos((1.0|_div_|x[0]-_log_(x[0])+x[0]*1.0*sin(x[0]))))|_div_|cos((x[0]|_div_|(1.0+1.0)-x[0]|_div_|_log_(cos(1.0)|_div_|(1.0*x[0]))))))+sin((cos(_exp_((1.0|_div_|1.0))|_div_|(_exp_(1.0)+1.0*x[0]))*_log_(cos(x[0]-1.0|_div_|_log_(1.0))))-x[0]*x[0]*cos(((_log_((x[0]|_div_|1.0))-((x[0]*1.0)*(x[0]|_div_|1.0)))+_log_((sin(1.0)+1.0+1.0)))-x[0]*cos(1.0))+cos(1.0))|_div_|x[0])))|_div_|(1.0|_div_|((_log_((cos(_exp_(((_exp_(x[0]*_log_((_log_(x[0])+_exp_(1.0))))-cos(cos(sin((x[0]*_log_(1.0)))))-(_log_(x[0])+_log_(1.0)))-_log_((_log_(x[0]-_exp_(x[0])+1.0)+x[0]))-((1.0*((1.0-1.0)+(x[0]*1.0))-1.0-_exp_(sin(cos(x[0]))))|_div_|cos(_exp_(_exp_(_log_(_log_(x[0])))))))))-_exp_((sin(_log_(_log_(x[0]))|_div_|(1.0-x[0]|_div_|1.0))|_div_|1.0)-x[0])*(_exp_(_log_(x[0]))+1.0|_div_|1.0-(x[0]-x[0]-_exp_(x[0])*(((x[0]-(1.0+1.0))|_div_|((x[0]|_div_|1.0)*sin(x[0])))+sin(x[0])+x[0])+x[0])|_div_|(sin(x[0]-1.0*1.0*1.0+1.0)*(x[0]-(cos(x[0])+sin((1.0-1.0))))-sin(_log_(_log_(sin((x[0]-x[0]))))))-cos(x[0]|_div_|_log_(_exp_(x[0]))+(_log_(1.0)|_div_|cos(x[0]))*x[0]|_div_|_log_(1.0)|_div_|_exp_(1.0-1.0))-(1.0-sin(sin(((1.0|_div_|x[0])+1.0-x[0])))-(x[0]-(cos(sin(1.0))+(x[0]-1.0+(1.0|_div_|x[0]))))|_div_|1.0*_exp_((1.0*1.0+x[0]*cos((x[0]+x[0]))))*x[0]))))*1.0)+_log_(sin(x[0]))|_div_|x[0]|_div_|_exp_(_log_(sin((((cos((x[0]|_div_|1.0*(_log_(x[0])|_div_|cos((x[0]+x[0])))))*sin(((sin(x[0])-(1.0*x[0]))*(_exp_(1.0)-sin(1.0)))*(_log_((1.0|_div_|1.0))*_exp_(cos(x[0]))))+x[0]+x[0]|_div_|1.0+(x[0]-1.0|_div_|(1.0+x[0])*(x[0]|_div_|1.0)))-1.0)-((_log_(1.0)+(_log_(sin(1.0))-x[0]))|_div_|(x[0]-(sin(x[0])+1.0*x[0]-1.0))+cos((1.0*sin((x[0]-x[0])))+1.0+sin((sin(1.0)+x[0])*_log_(cos(1.0))))))|_div_|(x[0]*_log_(cos(_log_(x[0]*_log_(x[0]-x[0]))+1.0))|_div_|cos(x[0])))))))', 20), "Error")


if __name__ == '__main__':
    unittest.main()
