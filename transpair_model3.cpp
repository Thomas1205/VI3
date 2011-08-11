/*

EGYPT Toolkit for Statistical Machine Translation
Written by Yaser Al-Onaizan, Jan Curin, Michael Jahr, Kevin Knight, John Lafferty, Dan Melamed, David Purdy, Franz Och, Noah Smith, and David Yarowsky.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, 
USA.

*/
/*--
transpair_model3: representation of a translation pair for model3 training
allowing for fast access (esp. to t table).

Franz Josef Och (30/07/99)
--*/
#include "transpair_model3.h"
#include <algorithm>


/******** added by Thomas Schoenemann *********/
#include "ClpSimplex.hpp"
#include "CbcModel.hpp"
#include "OsiClpSolverInterface.hpp"

#include "CglGomory/CglGomory.hpp"

#include "CbcHeuristic.hpp"
#include "CbcBranchActual.hpp"
/**********************************************/

#define ENABLE_SOS

transpair_model3::transpair_model3(const Vector<WordIndex>&es, const Vector<WordIndex>&fs, tmodel<COUNT, PROB>&tTable, amodel<PROB>&aTable, amodel<PROB>&dTable, nmodel<PROB>&nTable, double _p1, double _p0, void*)
  : transpair_model2(es,fs,tTable,aTable),d(es.size(), fs.size()),n(es.size(), MAX_FERTILITY+1), p0(_p0), p1(_p1)
{ 
  WordIndex l=es.size()-1,m=fs.size()-1;
  for(WordIndex i=0;i<=l;i++)
    {
      for(WordIndex j=1;j<=m;j++)
	d(i, j)=dTable.getValue(j, i, l, m);
      if( i>0 )
	{
	  for(WordIndex f=0;f<MAX_FERTILITY;f++)
	    n(i, f)=nTable.getValue(es[i], f);
	  n(i,MAX_FERTILITY)=PROB_SMOOTH;
	}
    }
}

LogProb transpair_model3::scoreOfMove(const alignment&a, WordIndex new_i, WordIndex j, double,bool forModel3)const
{
  LogProb change;
  const WordIndex old_i=a(j);
  WordIndex f0=a.fert(0);
  if (old_i == new_i)
    change=1.0;
  else if (old_i == 0)
    change=((double)p0*p0/p1) *
      (( (DeficientDistortionForEmptyWord?(max(2,int(m))/DeficientDistortionForEmptyWord):f0)*(m-f0+1.0)) / ((m-2*f0+1)*(m-2*f0+2.0))) *
      ((PROB)(forModel3?(a.fert(new_i)+1.0):1.0)) *
      (get_fertility(new_i, a.fert(new_i)+1) / get_fertility(new_i, a.fert(new_i)))*
      (t(new_i, j)/t(old_i, j))*
      (forModel3?d(new_i, j):1.0);
  else if (new_i == 0)
    change=(double(p1) / (p0*p0)) *
      (double((m-2*f0)*(m-2*f0-1))/( (DeficientDistortionForEmptyWord?(max(2,int(m))/DeficientDistortionForEmptyWord):(1+f0))*(m-f0))) *
      (forModel3?(1.0/a.fert(old_i)):1.0) *
      (get_fertility(old_i, a.fert(old_i)-1) /get_fertility(old_i, a.fert(old_i)))*
      (t(new_i, j) /t(old_i, j)) *
      (forModel3?(1.0 / d(old_i, j)):1.0);
  else
    change=(forModel3?((a.fert(new_i)+1.0)/a.fert(old_i)):1.0) *
      (get_fertility(old_i,a.fert(old_i)-1) / get_fertility(old_i,a.fert(old_i))) *
      (get_fertility(new_i,a.fert(new_i)+1) /get_fertility(new_i,a.fert(new_i))) *
      (t(new_i,j)/t(old_i,j)) *
      (forModel3?(d(new_i,j)/d(old_i,j)):1.0);
  return change;
}

LogProb transpair_model3::scoreOfSwap(const alignment&a, WordIndex j1, WordIndex j2, double,bool forModel3)const 
{
  PROB score=1;
  assert(j1<j2);
  WordIndex i1=a(j1), i2=a(j2);
  if (i1!=i2) 
    {
      score=(t(i2, j1)/t(i1, j1))*(t(i1, j2)/t(i2, j2));
      if( forModel3 )
	{
	  if (i1)
	    score *= d(i1, j2)/d(i1, j1);
	  if (i2)
	    score *= d(i2, j1)/d(i2, j2);    
	}
    }
  return score;
}

ostream&operator<<(ostream&out, const transpair_model3&m)
{
  for(WordIndex i=0;i<=m.get_l();i++)
    {
      out << "EF-I:"<<i<<' ';
      for(WordIndex j=1;j<=m.get_m();j++)
	out << "("<<m.t(i,j)<<","<<m.d(i,j)<<")";
      for(WordIndex j=1;j<MAX_FERTILITY;j++)
	if( i>0 )
	  out << "(fert:"<<m.get_fertility(i,j)<<")";
      out << '\n';
    }
  out << "T:" << m.t << "D:" << m.d << "A:" << m.a  << "N:" << m.n << m.p0 << m.p1 << '\n';
  return out;
}

LogProb transpair_model3::_scoreOfMove(const alignment&a, WordIndex new_i, WordIndex j,double)const
{
  alignment b(a);
  b.set(j, new_i);
  LogProb a_prob=prob_of_target_and_alignment_given_source(a);
  LogProb b_prob=prob_of_target_and_alignment_given_source(b);
  if( a_prob )
    return b_prob/a_prob;
  else if( b_prob )
    return 1e20;
  else
    return 1.0;
}

LogProb transpair_model3::_scoreOfSwap(const alignment&a, WordIndex j1, WordIndex j2,double thisValue)const
{
  alignment b(a);
  b.set(j1, a(j2));
  b.set(j2, a(j1));
  LogProb a_prob=thisValue;
  if( a_prob<0.0 )
    a_prob=prob_of_target_and_alignment_given_source(a);
  massert(a_prob==prob_of_target_and_alignment_given_source(a));
  LogProb b_prob=prob_of_target_and_alignment_given_source(b);
  if( a_prob )
    return b_prob/a_prob;
  else if( b_prob )
    return 1e20;
  else
    return 1.0;
}

LogProb transpair_model3::prob_of_target_and_alignment_given_source(const alignment&al,bool verb)const
{
  LogProb total = 1.0 ;
  static const LogProb zero = 1E-299 ; 
  total *= pow(double(1-p1), m-2.0 * al.fert(0)) * pow(double(p1), double(al.fert(0)));
  if( verb) cerr << "IBM-3: (1-p1)^(m-2 f0)*p1^f0: " << total << '\n';
  for (WordIndex i = 1 ; i <= al.fert(0) ; i++)
    total *= double(m - al.fert(0) - i + 1) / (double(DeficientDistortionForEmptyWord?(max(2,int(m))/DeficientDistortionForEmptyWord):i)) ;
  if( verb) cerr << "IBM-3: +NULL:binomial+distortion " << total << '\n';
  for (WordIndex i = 1 ; i <= l ; i++)
    {
      total *= get_fertility(i, al.fert(i)) * (LogProb) factorial(al.fert(i));
      if( verb) cerr << "IBM-3: fertility of " << i << " with factorial " << get_fertility(i, al.fert(i)) * (LogProb) factorial(al.fert(i)) << " -> " << total << '\n';
    }
  for (WordIndex j = 1 ; j <= m ; j++)
    {
      total*= get_t(al(j), j) ;
      massert( get_t(al(j), j)>=PROB_SMOOTH );
      if( verb) cerr << "IBM-3: t of " << j << " " << al(j) << ": " << get_t(al(j), j)  << " -> " << total << '\n';
      if (al(j))
	{
	  total *= get_d(al(j), j);
	  if( verb) cerr << "IBM-3: d of " << j << ": " << get_d(al(j), j)  << " -> " << total << '\n';
	}
    }
  return total?total:zero;
}


void transpair_model3::computeScores(const alignment&al,vector<double>&d)const
{
  LogProb total1 = 1.0,total2=1.0,total3=1.0,total4=1.0 ;
  total1 *= pow(double(1-p1), m-2.0 * al.fert(0)) * pow(double(p1), double(al.fert(0)));
  for (WordIndex i = 1 ; i <= al.fert(0) ; i++)
    total1 *= double(m - al.fert(0) - i + 1) / (double(DeficientDistortionForEmptyWord?(max(2,int(m))/DeficientDistortionForEmptyWord):i)) ;
  for (WordIndex i = 1 ; i <= l ; i++)
    {
      total2 *= get_fertility(i, al.fert(i)) * (LogProb) factorial(al.fert(i));
    }
  for (WordIndex j = 1 ; j <= m ; j++)
    {
      total3*= get_t(al(j), j) ;
      massert( get_t(al(j), j)>=PROB_SMOOTH );
      if (al(j))
	{
	  total4 *= get_d(al(j), j);
	}
    }
  d.push_back(total1);//5
  d.push_back(total2);//6
  d.push_back(total3);//7
  d.push_back(total4);//8
}


/************ added by Thomas Schoenemann ******************/


class GIZAIBM3IPHeuristic : public CbcHeuristic {
public:

  GIZAIBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nNullFertilityVars, uint nFertilityVarsPerTargetWord);

  //useful for the clone()-rourine
  GIZAIBM3IPHeuristic(const CbcHeuristic& heuristic, uint I, uint J, uint nNullFertilityVars, uint nFertilityVarsPerTargetWord);

  ~GIZAIBM3IPHeuristic();

  virtual CbcHeuristic* clone() const;

  virtual void resetModel(CbcModel* model);

  virtual int solution(double& objectiveValue, double* newSolution);

  virtual bool shouldHeurRun(int whereFrom);

protected:

  uint I_;
  uint J_;
  uint nNullFertilityVars_;
  uint nFertilityVarsPerTargetWord_;
  uint* fert_count_;
};


GIZAIBM3IPHeuristic::GIZAIBM3IPHeuristic(CbcModel& model, uint I, uint J, uint nNullFertilityVars, uint nFertilityVarsPerTargetWord) :
  CbcHeuristic(model), I_(I), J_(J), nNullFertilityVars_(nNullFertilityVars), nFertilityVarsPerTargetWord_(nFertilityVarsPerTargetWord)
{ fert_count_ = new uint[I_+1]; }

GIZAIBM3IPHeuristic::GIZAIBM3IPHeuristic(const CbcHeuristic& heuristic, uint I, uint J, uint nNullFertilityVars, uint nFertilityVarsPerTargetWord) :
  CbcHeuristic(heuristic), I_(I), J_(J), nNullFertilityVars_(nNullFertilityVars), nFertilityVarsPerTargetWord_(nFertilityVarsPerTargetWord)
{ fert_count_ = new uint[I_+1]; }

GIZAIBM3IPHeuristic::~GIZAIBM3IPHeuristic() {
  delete[] fert_count_;
}

/*virtual*/ CbcHeuristic* GIZAIBM3IPHeuristic::clone() const {
  return new GIZAIBM3IPHeuristic(*this,I_,J_,nNullFertilityVars_,nFertilityVarsPerTargetWord_);
}

/*virtual*/ void GIZAIBM3IPHeuristic::resetModel(CbcModel* /*model*/) {
  assert(false);
}

/*virtual*/ bool GIZAIBM3IPHeuristic::shouldHeurRun(int /*whereFrom*/) {

   const int depth = model_->currentDepth();
   const int pass = model_->getCurrentPassNumber();

   return (depth <= 10 || pass == 1);
}

/*virtual*/ int GIZAIBM3IPHeuristic::solution(double& objectiveValue, double* newSolution) {

  const OsiSolverInterface* solver = model_->solver();
  const double* cur_solution = solver->getColSolution();

  const uint nVars = solver->getNumCols();

  for (uint v=0; v < nVars; v++) {
    newSolution[v] = 0.0;
  }

  for (uint i=0; i < I_+1; i++)
    fert_count_[i] = 0;

  for (uint j=0; j < J_; j++) {

    double max_var = 0.0;
    uint aj = 100000;

    for (uint i=0; i <= I_; i++) {

      double var_val = cur_solution[j*(I_+1)+i];

      if (var_val > max_var) {
	max_var = var_val;
	aj = i;
      }
    }

    newSolution[j*(I_+1)+aj] = 1.0;

    fert_count_[aj]++;
  }
  
  const uint fert_var_offs = J_*(I_+1);

  for (uint i=0; i <= I_; i++) {

    uint idx = fert_var_offs + fert_count_[i];
    if (i > 0)
      idx += nNullFertilityVars_ + (i-1)* nFertilityVarsPerTargetWord_;

    newSolution[idx] = 1.0;
  }

  uint return_code = 0;

  const double* objective = solver->getObjCoefficients();

  double new_energy = 0.0;
  for (uint v=0; v < nVars; v++)
    new_energy += newSolution[v] * objective[v];

  if (new_energy < objectiveValue) {

    return_code = 1;
  }

  //std::cerr << "new integer energy " << new_energy << ",  previous " << objectiveValue << std::endl;

  //delete[] fert_count;

  return return_code;
}


class IBM3ExclusionCutGenerator : public CglCutGenerator {
public:

  IBM3ExclusionCutGenerator(const CbcModel& cbc_model, const double* cost,
			    const double* jcost_lower_bound_, const double* icost_lower_bound, const double* ibound2,
			    const Array2<double>& approx_icost, 
			    uint curJ, uint curI, uint nNullFertVars, uint nFertVarsPerTargetWord, double lower_bound,
			    double loose_lower_bound, double loose_lower_bound2);

  virtual void generateCuts(const OsiSolverInterface& si, OsiCuts & cs,
			    const CglTreeInfo = CglTreeInfo()) const;
  
  virtual CglCutGenerator* clone() const;

protected:

  uint curJ_;
  uint curI_;
  uint nNullFertVars_;
  uint nFertVarsPerTargetWord_;

  double lower_bound_;
  double loose_lower_bound_;
  double loose_lower_bound2_;

  const CbcModel& cbc_model_;
  const double* cost_;
  const double* jcost_lower_bound_;
  const double* icost_lower_bound_;
  const double* ibound2_;
  const Array2<double>& approx_icost_;
};


IBM3ExclusionCutGenerator::IBM3ExclusionCutGenerator(const CbcModel& cbc_model, const double* cost,
						     const double* jcost_lower_bound, 
						     const double* icost_lower_bound, const double* ibound2,
						     const Array2<double>& approx_icost, 
						     uint curJ, uint curI, uint nNullFertVars, uint nFertVarsPerTargetWord, 
						     double lower_bound, double loose_lower_bound, double loose_lower_bound2) :
  curJ_(curJ), curI_(curI), nNullFertVars_(nNullFertVars), nFertVarsPerTargetWord_(nFertVarsPerTargetWord), 
  lower_bound_(lower_bound), loose_lower_bound_(loose_lower_bound),
  loose_lower_bound2_(loose_lower_bound2), cbc_model_(cbc_model), cost_(cost), jcost_lower_bound_(jcost_lower_bound),
  icost_lower_bound_(icost_lower_bound), ibound2_(ibound2), approx_icost_(approx_icost)
{}

/*virtual*/ CglCutGenerator* IBM3ExclusionCutGenerator::clone() const {

  return new IBM3ExclusionCutGenerator(cbc_model_, cost_, jcost_lower_bound_, icost_lower_bound_, ibound2_, approx_icost_,
				       curJ_, curI_, nNullFertVars_, nFertVarsPerTargetWord_, lower_bound_, loose_lower_bound_, 
				       loose_lower_bound2_);
}


/*virtual*/ void IBM3ExclusionCutGenerator::generateCuts(const OsiSolverInterface& /*si*/, OsiCuts & cs,
							 const CglTreeInfo info) const {

  if (info.pass == 1) {

    const double upper_bound = cbc_model_.getObjValue();
    const double gap = upper_bound - lower_bound_;
    const double loose_gap = upper_bound - loose_lower_bound_;
    const double loose_gap2 = upper_bound - loose_lower_bound2_;

    const uint fert_var_offs = curJ_*(curI_+1);

    const double var_limit = 0.0;

    const double* colUpper = cbc_model_.getColUpper();

    for (uint j=0; j < curJ_; j++) {
    
      for (uint aj=0; aj <= curI_; aj++) {

	const int idx = j*(curI_+1) + aj;

	if ( colUpper[idx] > 0.0 && cost_[idx] >= gap+jcost_lower_bound_[j]+0.1) {
	 
	  OsiColCut new_cut;
	  new_cut.setLbs(1,&idx,&var_limit);
	  new_cut.setUbs(1,&idx,&var_limit);

	  cs.insert(new_cut);
	}
      }
    }

    for (uint i=0; i <= curI_; i++) {

      uint cur_limit = (i == 0) ? nNullFertVars_ : nFertVarsPerTargetWord_;
      
      for (uint f=0; f < cur_limit; f++) {

	int idx = fert_var_offs + f;
	if (i > 0)
	  idx += nNullFertVars_ + (i-1) * nFertVarsPerTargetWord_;

	if (colUpper[idx] > 0.0) {
	
	  if ( (cost_[idx] >= icost_lower_bound_[i] + loose_gap + 0.1)
	       || ((approx_icost_(i,f) >= loose_gap2 + 0.1 + ibound2_[i])) ) {
	       //|| ((cost_[idx] >= loose_gap2 + 0.1 + ibound2_[i])) ) {

	    OsiColCut new_cut;
	    new_cut.setLbs(1,&idx,&var_limit);
	    new_cut.setUbs(1,&idx,&var_limit);
	    
	    cs.insert(new_cut);
	  }
	}
      }
    }
  }
}


class IBM3CountCutGenerator : public CglCutGenerator {
public:

  IBM3CountCutGenerator( uint curJ, uint curI, uint nNullFertVars, uint nFertVarsPerTargetWord);

  virtual void generateCuts(const OsiSolverInterface& si, OsiCuts & cs,
			    const CglTreeInfo = CglTreeInfo()) const;
  
  virtual CglCutGenerator* clone() const;

protected:

  uint curJ_;
  uint curI_;
  uint nNullFertVars_;
  uint nFertVarsPerTargetWord_;
};

IBM3CountCutGenerator::IBM3CountCutGenerator(uint curJ, uint curI, uint nNullFertVars, uint nFertVarsPerTargetWord)
  : curJ_(curJ), curI_(curI), nNullFertVars_(nNullFertVars), nFertVarsPerTargetWord_(nFertVarsPerTargetWord) {
}

/*virtual*/ CglCutGenerator* IBM3CountCutGenerator::clone() const {

  return new IBM3CountCutGenerator(curJ_,curI_,nNullFertVars_,nFertVarsPerTargetWord_);
}

/*virtual*/ void IBM3CountCutGenerator::generateCuts(const OsiSolverInterface& si, OsiCuts & cs,
						     const CglTreeInfo) const {

  const double* cur_lp_solution = si.getColSolution();

  uint nVars = (curI_+1)*curJ_  //alignment variables
    + nNullFertVars_ + curI_*nFertVarsPerTargetWord_; //fertility variables
  //+ (curI_+1)*nFertVarsPerWord_; //fertility variables

  if (si.getNumCols() != (int) nVars) {
    std::cerr << "Warning: mismatch in the number of variables" << std::endl;
  }

  const uint fert_var_offs = (curI_+1)*curJ_;

  double* align_var_values = new double[curJ_];
  uint* align_var_indices = new uint[curJ_];

  int* col_idx = new int[2*std::max(nNullFertVars_,nFertVarsPerTargetWord_)+2];
  double* coeff = new double[2*std::max(nNullFertVars_,nFertVarsPerTargetWord_)+2];

  for (uint i=0; i <= curI_; i++) {

    bool values_initialized = false;

    int cur_limit = (i == 0) ? nNullFertVars_ : nFertVarsPerTargetWord_;

    for (int n=0; n < cur_limit-1; n++) {

      //a) forward cut

      uint idx = fert_var_offs + n;
      if (i > 0)
	idx += nNullFertVars_ + (i-1) * nFertVarsPerTargetWord_;
      
      double var_val = cur_lp_solution[idx];
	      
      //NOTE: if var_val == 0, the cut inequality must be fullfilled
      if (var_val > 0.01 && var_val < 0.99) {

	if (!values_initialized) {

	  for (uint j=0; j < curJ_; j++) {

	    align_var_values[j] = cur_lp_solution[j*(curI_+1)+i];
	    align_var_indices[j] = j*(curI_+1)+i;

	    if (align_var_indices[j] >= nVars) {
	      std::cerr << " error in line " << __LINE__ << std::endl;
	    }
	  }

	  //bubble sort (highest values go in front)
	  for (uint k1 = 0; k1 < curJ_-1; k1++) {
	    for (uint k2 = 0; k2 < curJ_-1-k1; k2++) {
	      
	      if (align_var_values[k2] < align_var_values[k2+1]) {
		
		std::swap(align_var_values[k2],align_var_values[k2+1]);
		std::swap(align_var_indices[k2],align_var_indices[k2+1]);
	      }
	    }
	  }

	  values_initialized = true;
	}

	double fwd_sum = var_val;
	for (int nn = 0; nn < n; nn++) {

	  //fwd_sum += cur_lp_solution[ fert_var_offs + i*nFertVarsPerWord_ + nn];
	  fwd_sum += cur_lp_solution[ idx - n + nn];
	}

	for (uint j=0; j <= (uint) n; j++) {
	  fwd_sum += align_var_values[j];
	}

	if (fwd_sum > n + 1.01) {

	  //std::cerr << "adding cut" << std::endl;

	  for (int k=0; k < 2*n+2; k++)
	    coeff[k] = 1.0;

	  for (int k=0; k <= n; k++)
	    col_idx[k] = idx - n + k; //fert_var_offs + i*nFertVarsPerWord_ + k;
	  
	  for (int k=0; k <= n; k++) {

	    //DEBUG
	    if ( align_var_indices[k] >= nVars) {
	      std::cerr << " error in line " << __LINE__ << std::endl;

	      std::cerr << "offending index: " << align_var_indices[k] << std::endl;
	      std::cerr << "k = " << k << std::endl;
	      std::cerr << "J = " << curJ_ << std::endl;	      
	    }
	    //END_DEBUG

	    col_idx[n+1 + k] = align_var_indices[k];
	  }

	  OsiRowCut newCut;
	  newCut.setRow(2*n+2,col_idx,coeff,false);
	  newCut.setLb(0.0);
	  newCut.setUb(n+1);   
	  
	  cs.insert(newCut);  
	}
      }
      
#if 1
      //b) backward cut
      uint bwd_idx = fert_var_offs + nNullFertVars_ + i*nFertVarsPerTargetWord_ - n - 1;

      //double bwd_var_val = cur_lp_solution[ fert_var_offs + (i+1)*nFertVarsPerWord_ - n - 1];
      double bwd_var_val = cur_lp_solution[ bwd_idx ];
      
      if (bwd_var_val > 0.01 && bwd_var_val < 0.99) {

	if (!values_initialized) {

	  for (uint j=0; j < curJ_; j++) {

	    align_var_values[j] = cur_lp_solution[j*(curI_+1)+i];
	    align_var_indices[j] = j*(curI_+1)+i;
	  }

	  //bubble sort (highest values go in front)
	  for (uint k1 = 0; k1 < curJ_-1; k1++) {
	    for (uint k2 = 0; k2 < curJ_-1-k1; k2++) {
	      
	      if (align_var_values[k2] < align_var_values[k2+1]) {
		
		std::swap(align_var_values[k2],align_var_values[k2+1]);
		std::swap(align_var_indices[k2],align_var_indices[k2+1]);
	      }
	    }
	  }

	  values_initialized = true;
	}

	double bwd_sum = var_val;
	for (int nn = 0; nn < n; nn++)
	  //bwd_sum += cur_lp_solution[ fert_var_offs + (i+1)*nFertVarsPerWord_ - nn - 1];
	  bwd_sum += cur_lp_solution[fert_var_offs + nNullFertVars_ + i*nFertVarsPerTargetWord_ - nn - 1];

	for (uint j=0; j <= (uint) n; j++)
	  bwd_sum += align_var_values[j];

	if (bwd_sum > n + 1.01) {

	  //std::cerr << "adding reverse cut" << std::endl;
	  
	  for (int k=0; k <= n; k++)
	    coeff[k] = 1.0;
	  for (int k=0; k <= n; k++)
	    coeff[n+1 + k] = -1.0;

	  for (int k=0; k <= n; k++)
	    //col_idx[k] = fert_var_offs + (i+1)*nFertVarsPerWord_ - k - 1;
	    col_idx[k] = fert_var_offs + nNullFertVars_ + i*nFertVarsPerTargetWord_ - k - 1;
	  
	  for (int k=0; k <= n; k++)
	    col_idx[n+1 + k] = align_var_indices[k];

	  OsiRowCut newCut;
	  newCut.setRow(2*n+2,col_idx,coeff,false);
	  newCut.setLb(-1e20);
	  newCut.setUb(0.0);   
	  
	  cs.insert(newCut);  
	}
	
      }
#endif
    }    
  }

  delete[] align_var_values;
  delete[] align_var_indices;
  delete[] col_idx;
  delete[] coeff;
}

void transpair_model3::compute_viterbi_alignment(alignment& al, double upper_log_bound,
						 const alignment& upper_bound_alignment) {


  //std::cerr << "deriving Viterbi alignment" << std::endl;

  const uint curI = l;
  const uint curJ = m;

  //std::cerr << "curI: " << curI << ", curJ: " << curJ << std::endl;

  double* values = new double[curJ];

  uint* wmax_fert = new uint[curI+1];
  for (uint i=0; i <= curI; i++)
    wmax_fert[i] = curJ;

//   uint max_fertility = std::max<uint>(15,curJ/2);
//   if (2*l < m)
//     max_fertility = std::max<uint>(max_fertility,25);

  //uint max_fertility = std::max<uint>(MAX_FERTILITY, curJ/2);

  //uint nFertVarsPerWord = std::min(curJ,max_fertility) + 1;

  uint nNullFertVars = curJ/2 + 1;

  uint nFertVarsPerTargetWord = std::min(curJ,MAX_FERTILITY) + 1;

  //std::cerr << "computing viterbi alignment" << std::endl;

  uint nVars = (curI+1)*curJ  //alignment variables
    + nNullFertVars + curI*nFertVarsPerTargetWord;
    //+ (curI+1)*nFertVarsPerWord; //fertility variables
  
  uint fert_var_offs = (curI+1)*curJ;
  //std::cerr << "fert_var_offs: " << fert_var_offs << std::endl;

  uint nConstraints = curJ // alignment variables must sum to 1 for each source word
    + curI + 1 //fertility variables must sum to 1 for each target word including the empty word
    + curI + 1; //fertility variables must be consistent with alignment variables

  uint fert_con_offs = curJ;
  uint consistency_con_offs = fert_con_offs + curI + 1;

  double* cost = new double[nVars];
  double* var_lb = new double[nVars];
  double* var_ub = new double[nVars];

  for (uint v=0; v < nVars; v++) {
    
    cost[v] = 1000.0;
    var_lb[v] = 0.0;
    var_ub[v] = 1.0;
  }

  double* jcost_lower_bound = new double[curJ];
  double* icost_lower_bound = new double[curI+1];

  std::vector< std::set<uint> > given_alignment;
  given_alignment.resize(curI+1);

  for (uint j=0; j < curJ; j++) {
    uint i = upper_bound_alignment.get_al(j);
    given_alignment[i].insert(j);
  }

  //set cost for the alignment variables
  for (uint j=0; j < curJ; j++) {

    const uint cur_offs = j*(curI+1);

    double min_cost = 1e50;
    uint arg_min = 0;

    for (uint i=0; i <= curI; i++) {

      double total = get_t(i, j+1) ;
      if (i)
	{
	  total *= get_d(i, j+1);
	}

      cost[cur_offs + i] = -log(total);

      if (total < min_cost) {
	min_cost = total;
	arg_min = i;
      }
    }

    jcost_lower_bound[j] = min_cost;
  }

  //std::cerr << "deficient empty word: " << DeficientDistortionForEmptyWord << std::endl;

  //set cost for the fertility variables of the empty word
  double min_empty_fert_cost = 1e20;
  for (uint fert=0; fert < nNullFertVars /*nFertVarsPerWord*/; fert++) {

    if (curJ-fert >= fert) {

      long double prob = 1.0;
      prob *= pow(double(1-p1), m-2.0 * fert) * pow(double(p1), double(fert));
      for (WordIndex i = 1 ; i <= fert ; i++)
	prob *= double(m - fert - i + 1) / (double(DeficientDistortionForEmptyWord?(max(2,int(m))/DeficientDistortionForEmptyWord):i)) ;
            
      cost[fert_var_offs + fert] = - logl( prob );

      assert(!isnan(cost[fert_var_offs+fert]));

      if (cost[fert_var_offs + fert] < min_empty_fert_cost) {
 	min_empty_fert_cost = cost[fert_var_offs + fert];
      }
    }
    else {      
      cost[fert_var_offs + fert] = 1e10;
      var_ub[fert_var_offs + fert] = 0.0;
    }
  }
  icost_lower_bound[0] = min_empty_fert_cost;

  for (uint i=0; i < curI; i++) {

    double min_cost = 1e50;

    //for (uint fert=0; fert < nFertVarsPerWord; fert++) {
    for (uint fert=0; fert < nFertVarsPerTargetWord; fert++) {
      
      //uint idx = fert_var_offs + (i+1)*nFertVarsPerWord + fert;
      uint idx = fert_var_offs + nNullFertVars + i*nFertVarsPerTargetWord + fert;
      
      if (fert > MAX_FERTILITY)
	cost[idx] = 1e10;
      else
	cost[idx] = -logl( get_fertility(i+1, fert) * (LogProb) factorial(   fert  ) );
      
      assert(!isnan(cost[fert_var_offs + (i+1)*nFertVarsPerWord + fert]));

      if (cost[idx] < min_cost)
	min_cost = cost[idx];
    }
    
    icost_lower_bound[i+1] = min_cost;
  }
  
  Array2<double> ifert_cost(curJ+1,curI+1);
  for (uint j=0; j<= curJ; j++)
    for (uint i=0; i <= curI; i++)
      ifert_cost(j,i) = 1e50;
  
  for (uint f=0; f < nNullFertVars /*nFertVarsPerWord*/ ; f++) {
    ifert_cost(f,0) = cost[fert_var_offs + f];
  }

  for (uint i = 1; i <= curI; i++) {

    for (uint j=0; j <= curJ; j++) {

      const uint offs = fert_var_offs+nNullFertVars + (i-1)*nFertVarsPerTargetWord;

      //double opt_cost = ifert_cost(j,i-1) + cost[fert_var_offs+i*nFertVarsPerWord];
      double opt_cost = ifert_cost(j,i-1) + cost[offs];

      for (uint f=1; f < std::min(j+1,nFertVarsPerTargetWord); f++) {
	
	//double hyp_cost = ifert_cost(j-f,i-1) + cost[fert_var_offs+i*nFertVarsPerWord + f]; 
	double hyp_cost = ifert_cost(j-f,i-1) + cost[offs + f];

	if (hyp_cost < opt_cost) {
	  opt_cost = hyp_cost;
	}
      }

      ifert_cost(j,i) = opt_cost;
    }
  }

  Array2<double> bwd_ifert_cost(curJ+1,curI+1);
  for (uint j=0; j<= curJ; j++)
    for (uint i=0; i <= curI; i++)
      bwd_ifert_cost(j,i) = 1e50;

  for (uint f=0; f < nFertVarsPerTargetWord ; f++) {
    bwd_ifert_cost(f,curI) = cost[fert_var_offs+nNullFertVars + (curI-1)*nFertVarsPerTargetWord + f];
  }
  for (int i=curI-1; i >= 1; i--) {
    
    for (uint j=0; j <= curJ; j++) {

      const uint offs = fert_var_offs+nNullFertVars + (i-1)*nFertVarsPerTargetWord;

      double opt_cost = bwd_ifert_cost(j,i+1) + cost[offs];
    
      for (uint f=1; f < std::min(j+1,nFertVarsPerTargetWord); f++) {

	double hyp_cost = bwd_ifert_cost(j-f,i+1) + cost[offs + f];

	if (hyp_cost < opt_cost) {
	  opt_cost = hyp_cost;
	}
      }

      bwd_ifert_cost(j,i) = opt_cost;
    }
  }

  //the backward values for the empty word will never be used -> don't even compute them
  //empty word
//   for (uint j=0; j <= curJ; j++) {
//     double opt_cost = bwd_ifert_cost(j,1) + cost[fert_var_offs];

//     for (uint f=1; f < std::min(j+1,nNullFertVars); f++) {

//       double hyp_cost = bwd_ifert_cost(j-f,1) + cost[fert_var_offs+ f];
//       if (hyp_cost < opt_cost) {
// 	opt_cost = hyp_cost;
//       }
//     }
//     bwd_ifert_cost(j,0) = opt_cost;
//   }
  
  double lower_bound = ifert_cost(curJ,curI);

  double j_lower_bound = 0.0;
  for (uint j=0; j < curJ; j++)
    j_lower_bound += jcost_lower_bound[j];

  lower_bound += j_lower_bound;

  double loose_lower_bound = icost_lower_bound[0];
  for (uint i=1; i<= curI; i++)
    loose_lower_bound += icost_lower_bound[i];
  for (uint j=0; j < curJ; j++)
    loose_lower_bound += jcost_lower_bound[j];

  double gap = upper_log_bound + 0.1 - lower_bound;
  double loose_gap = upper_log_bound + 0.1 - loose_lower_bound;

  //Array2<double> approx_icost(curI+1,nFertVarsPerWord);
  Array2<double> approx_icost(curI+1,std::max(nNullFertVars,nFertVarsPerTargetWord));
  double* ibound2 = new double[curI+1];

  double loose_bound2 = 0.0;

  for (uint i=0; i <= curI; i++) {

    for (uint j=0; j < curJ; j++)
      values[j] = cost[j*(curI+1)+i] - jcost_lower_bound[j];

    std::sort(values,values+curJ);

    for (uint j=1; j < curJ; j++)
      values[j] += values[j-1];
   
    //approx_icost(i,0) = cost[fert_var_offs+i*nFertVarsPerWord];
    if (i == 0)
      approx_icost(i,0) = cost[fert_var_offs];
    else
      approx_icost(i,0) = cost[fert_var_offs + nNullFertVars + (i-1)*nFertVarsPerTargetWord];

    double min_cost = approx_icost(i,0);

    uint cur_limit = (i == 0) ? nNullFertVars : nFertVarsPerTargetWord;

    //for (uint f=1; f < nFertVarsPerWord; f++) {
    for (uint f=1; f < cur_limit; f++) {

      //approx_icost(i,f) = cost[fert_var_offs+i*nFertVarsPerWord+f] + values[f-1];
      if (i == 0)
	approx_icost(i,f) = cost[fert_var_offs+f] + values[f-1];
      else
	approx_icost(i,f) = cost[fert_var_offs + nNullFertVars + (i-1)*nFertVarsPerTargetWord + f] + values[f-1];

      if (approx_icost(i,f) < min_cost)
	min_cost = approx_icost(i,f);
    }
    ibound2[i] = min_cost;

    loose_bound2 += min_cost;
  }

  //std::cerr << "loose_bound: " << loose_lower_bound << std::endl;
  //std::cerr << "loose_bound2: " << loose_bound2 << std::endl;
  
  double loose_gap2 = upper_log_bound + 0.1 - loose_bound2;  

  uint nHighCost = 0;

  for (uint j=0; j < curJ; j++) {
    
    for (uint aj=0; aj <= curI; aj++) {
      
      if (cost[j*(curI+1) + aj] >= gap+jcost_lower_bound[j]+0.1) {

	var_ub[j*(curI+1) + aj] = 0.0;
	wmax_fert[aj]--;

	nHighCost++;
      }
    }
  }

  uint nFertExcluded = 0;
  for (uint i=0; i <= curI; i++) {

    uint cur_limit = (i == 0) ? nNullFertVars : nFertVarsPerTargetWord;

    for (uint f=0; f < cur_limit /*nFertVarsPerWord*/; f++) {

      uint idx = fert_var_offs + f;
      if (i>0)
	idx += nNullFertVars + (i-1) * nFertVarsPerTargetWord;

      if (f > wmax_fert[i]) {
 	var_ub[idx] = 0.0;
 	nFertExcluded++;
      }
      else if (cost[idx] >= icost_lower_bound[i] + loose_gap + 0.1 ) {
 	var_ub[idx] = 0.0;
 	nFertExcluded++;
      }
      //else if (cost[idx] >= loose_gap2 + 0.1 + ibound2[i]) {
      else if (approx_icost(i,f) >= loose_gap2 + 0.1 + ibound2[i]) { //CHANGED AFTER version 0.921
 	var_ub[idx] = 0.0;
 	nFertExcluded++;
      }
      else {

	double min_bound = 1e300;
	//uint lower = (i == curI) ? curJ-f : 0;
	//uint upper = (i == 0) ? 0 : curJ-f;
	//for (uint prev_j = lower; prev_j <= upper; prev_j++) {
	for (uint prev_j = 0; prev_j <= curJ-f; prev_j++) {


	  double hyp = 0.0;
	  if (i > 0)
	    hyp += ifert_cost(prev_j,i-1);
	  if (i < curI)
	    hyp += bwd_ifert_cost(curJ-prev_j-f,i+1);

	  if (hyp < min_bound)
	    min_bound = hyp;
	}

	min_bound += cost[idx] + j_lower_bound;

	if (min_bound >= upper_log_bound + 0.1) {
	  var_ub[idx] = 0.0;	  
	}
      }
    }
  }

  //std::cerr << nFertExcluded << " fert vars excluded" << std::endl;

  for (uint v=0; v < nVars; v++) {

    assert(!isnan(cost[v]));

    if (cost[v] > 1e10) {
      nHighCost++;
      cost[v] = 1e10;
      var_ub[v] = 0.0;
    }
  }
  //std::cerr << "nHighCost: " << nHighCost << std::endl;

  double* rhs_upper = new double[nConstraints];
  double* rhs_lower = new double[nConstraints];

  for (uint c=0; c < consistency_con_offs; c++) {
    rhs_upper[c] = 1.0;
    rhs_lower[c] = 1.0;
  }
  for (uint c=consistency_con_offs; c < nConstraints; c++) {
    rhs_upper[c] = 0.0;
    rhs_lower[c] = 0.0;
  }

  uint nMatrixEntries = (curI+1)*curJ // for the alignment unity constraints
    + nNullFertVars + curI * nFertVarsPerTargetWord // for the fertility unity constraints
    //+ (curI+1)*(nFertVarsPerWord+1) // for the fertility unity constraints
    //+ (curI+1)*(nFertVarsPerWord-1+curJ); // for the consistency constraints
    + (curI+1)*(std::max(nFertVarsPerTargetWord,nNullFertVars)-1+curJ); // for the consistency constraints //NOTE: currently a loose bound

  uint curMatrixEntry = 0;

  int* row_idx = new int[nMatrixEntries];
  int* col_idx = new int[nMatrixEntries];
  double* matrix_entry = new double[nMatrixEntries];

  //code unity constraints for alignment variables
  for (uint j=0; j < curJ; j++) {
  
    for (uint v= j*(curI+1); v < (j+1)*(curI+1); v++) {
      if (var_ub[v] > 0.0) {

	row_idx[curMatrixEntry] = j;
	col_idx[curMatrixEntry] = v;
	matrix_entry[curMatrixEntry] = 1.0;
	curMatrixEntry++;
      }
    }
  }

  //code unity constraints for fertility variables
  for (uint i=0; i <= curI; i++) {

    uint cur_limit = (i == 0) ? nNullFertVars : nFertVarsPerTargetWord;

    for (uint fert=0; fert < cur_limit /*nFertVarsPerWord*/; fert++) {

      //uint col = fert_var_offs + i*nFertVarsPerWord + fert;
      uint col = fert_var_offs + fert;
      if (i > 0)
	col += nNullFertVars + (i-1)*nFertVarsPerTargetWord;

      if (col == nVars -1 || var_ub[col] > 0.0) { //this is the easiest way to avoid segmentation faults

	row_idx[curMatrixEntry] = fert_con_offs + i;
	col_idx[curMatrixEntry] = col;
	matrix_entry[curMatrixEntry] = 1.0;
	curMatrixEntry++;
      }
    }
  }

  //code consistency constraints
  for (uint i=0; i <= curI; i++) {

    uint row = consistency_con_offs + i;

    uint cur_limit = (i == 0) ? nNullFertVars : nFertVarsPerTargetWord;

    for (uint fert=1; fert < cur_limit /*nFertVarsPerWord*/; fert++) {

      //uint col = fert_var_offs + i*nFertVarsPerWord + fert;
      uint col = fert_var_offs + fert;
      if (i > 0)
	col += nNullFertVars + (i-1)*nFertVarsPerTargetWord;

      
      if (var_ub[col] > 0.0) {

	row_idx[curMatrixEntry] = row;
	col_idx[curMatrixEntry] = col;
	matrix_entry[curMatrixEntry] = fert;
	curMatrixEntry++;
      }
    }

    for (uint j=0; j < curJ; j++) {

      uint col = j*(curI+1) + i;

      if (var_ub[col] > 0.0) {
	
	row_idx[curMatrixEntry] = row;
	col_idx[curMatrixEntry] = col;
	matrix_entry[curMatrixEntry] = -1.0;
	curMatrixEntry++;
      }
    }
  }

  if (curMatrixEntry > nMatrixEntries) {
    std::cerr << "ERROR: exceeded matrix entries" << std::endl;
  }

  CoinPackedMatrix coinMatrix(false, row_idx, col_idx, matrix_entry, curMatrixEntry);

  //std::cerr << "solving the ILP" << std::endl;

  OsiClpSolverInterface ilpSolver;
  ilpSolver.loadProblem(coinMatrix, var_lb, var_ub, cost, rhs_lower, rhs_upper);

  //std::cerr << "problem loaded" << std::endl;

  ilpSolver.messageHandler()->setLogLevel(0);
  ilpSolver.setLogLevel(0);

  if (((int) nVars) != ilpSolver.getNumCols() ) {

    std::cerr << "WARNING: mismatch in the number of variables" << std::endl;
  }

  //std::cerr << "setting integers" << std::endl;

  //for (uint v=fert_var_offs; v < nVars; v++) {
  for (uint v=0; v < (uint) ilpSolver.getNumCols(); v++) {

    if (var_ub[v] > 0.0) {
      ilpSolver.setInteger(v);
    }
  }
  
  ilpSolver.resolve();
  int error = 1 - ilpSolver.isProvenOptimal();

  //std::cerr << "LP relaxation solved" << std::endl;

  assert(!error);

  if (error) {
    std::cerr << "ERROR: solving the lp-relaxation failed. Exiting..." << std::endl;
    exit(1);
  }

  const double* lp_solution = ilpSolver.getColSolution();
  long double energy = 0.0;

  uint nNonIntegral = 0;
  uint nNonIntegralFert = 0;

  for (uint v=0; v < (uint) ilpSolver.getNumCols(); v++) {

    if (var_ub[v] > 0.0) {

      double var_val = lp_solution[v];
      
      if (var_val > 0.01 && var_val < 0.99) {
	nNonIntegral++;
	
	if (v >= fert_var_offs)
	  nNonIntegralFert++;
      }

      energy += cost[v] * var_val;
    }
  }

  //std::cerr << "initial lower bound: " << lower_bound << ", lp-relaxation energy: " << energy << std::endl;

//   std::cerr << nNonIntegral << " non-integral variables (" << (nNonIntegral - nNonIntegralFert)  
// 	    << "/" << nNonIntegralFert << ")" << std::endl;

  const double* solution = lp_solution;

  ilpSolver.setupForRepeatedUse();  

  CbcModel cbc_model(ilpSolver);

  if (nNonIntegral > 0) {

    cbc_model.messageHandler()->setLogLevel(0);

    double* best_sol = new double[nVars];
    uint* fert_count = new uint[curI+1];
    for (uint i=0; i<= curI; i++)
      fert_count[i] = 0;

    for (uint v=0; v < nVars; v++)
      best_sol[v] = 0.0;
    
    for (uint j=0; j < curJ; j++) {
      
      uint aj = upper_bound_alignment.get_al(j+1);
      fert_count[aj]++;
      best_sol[ j*(curI+1)+aj] = 1.0;
    } 
    for (uint i=0; i <= curI; i++) {
      
      uint idx = fert_var_offs + fert_count[i];
      if (i>0)
	idx += nNullFertVars + (i-1) * nFertVarsPerTargetWord;
      best_sol[idx] = 1.0;
      
      //best_sol[fert_var_offs + i*nFertVarsPerWord + fert_count[i] ] = 1.0;
    }
    
    double temp_energy = 0.0;
    for (uint v=0; v < (uint) ilpSolver.getNumCols(); v++)
      temp_energy += cost[v]*best_sol[v];

    //std::cerr << "setting best solution: " << best_sol << std::endl;

    cbc_model.setBestSolution(best_sol,ilpSolver.getNumCols(),temp_energy,true);

    //std::cerr << "set best solution: " << best_sol << std::endl;
    
    delete[] best_sol;
    delete[] fert_count;

    CglGomory gomory_cut;
    gomory_cut.setLimit(250);
    //gomory_cut.setLimit(750);
    gomory_cut.setAway(0.02);
    gomory_cut.setLimitAtRoot(250);
    //gomory_cut.setLimitAtRoot(750);
    gomory_cut.setAwayAtRoot(0.02);
    cbc_model.addCutGenerator(&gomory_cut,0,"Gomory Cut");

    GIZAIBM3IPHeuristic  ibm3_heuristic(cbc_model, curI, curJ, nNullFertVars, nFertVarsPerTargetWord);
    cbc_model.addHeuristic(&ibm3_heuristic,"GIZA IBM3 Heuristic");

#ifdef ENABLE_SOS
    CbcObject** sos = new CbcObject*[curJ + curI + 1];
    int* sos_idx = new int[std::max(curI+1,std::max(nFertVarsPerTargetWord, nNullFertVars))];
    double* sos_weight = new double[std::max(curI+1,std::max(nFertVarsPerTargetWord, nNullFertVars))];

    for (uint j=0; j < curJ; j++) {

      uint next = 0;

      for (uint i=0; i <= curI; i++) {	
	uint idx = j*(curI+1) + i;
	if (var_ub[idx] > 0.0) {
	  sos_idx[next] = idx;
	  sos_weight[next] = 1.0;
	  next++;
	}
      }

      CbcSOS* ptr = new CbcSOS(&cbc_model,next,sos_idx,sos_weight,j);
      ptr->setIntegerValued(true);

      sos[j] = ptr;
    }
    for (uint i=0; i <= curI; i++) {

      uint cur_limit = (i == 0) ? nNullFertVars : nFertVarsPerTargetWord;

      uint next = 0;

      for (uint f=0; f < cur_limit; f++) {

	uint idx = fert_var_offs + f;
	if (i > 0)
	  idx += nNullFertVars + (i-1) * nFertVarsPerTargetWord;

	if (var_ub[idx] > 0.0) {
	  sos_idx[next] = idx;
	  sos_weight[next] = 1.0;
	  next++;
	}
      }

      CbcSOS* ptr = new CbcSOS(&cbc_model,next,sos_idx,sos_weight,curJ+i);
      ptr->setIntegerValued(true);

      sos[curJ+i] = ptr;
    }

    delete[] sos_idx;
    delete[] sos_weight;

    cbc_model.addObjects(curJ+curI+1, sos);
#endif

    IBM3ExclusionCutGenerator excl_cg(cbc_model, cost, jcost_lower_bound, icost_lower_bound, ibound2,
 				      approx_icost, curJ, curI, nNullFertVars, nFertVarsPerTargetWord, lower_bound,
 				      loose_lower_bound, loose_bound2);
    cbc_model.addCutGenerator(&excl_cg,0,"Exclusion Cut");

    IBM3CountCutGenerator count_cg(curJ, curI, nNullFertVars, nFertVarsPerTargetWord);
    cbc_model.addCutGenerator(&count_cg,0,"Count Cut");

    cbc_model.branchAndBound();

#ifdef ENABLE_SOS
    for (uint k=0; k < curJ+curI+1; k++)
      delete sos[k];
    delete[] sos;
#endif

    if (!cbc_model.isProvenOptimal()) {

      std::cerr << "ERROR: the optimal solution could not be found. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isProvenInfeasible()) {

      std::cerr << "ERROR: problem marked as infeasible. Exiting..." << std::endl;
      exit(1);
    }
    if (cbc_model.isAbandoned()) {

      std::cerr << "ERROR: problem was abandoned. Exiting..." << std::endl;
      exit(1);
    }

    const double* cbc_solution = cbc_model.bestSolution();

    energy = 0.0;

    for (uint v=0; v < (uint) ilpSolver.getNumCols(); v++) {
      
      if (var_ub[v] > 0.0) {
	double var_val = cbc_solution[v];
	
	energy += cost[v] * var_val;
      }
    }

    solution = cbc_solution;
  }

  uint nNonIntegralVars = 0;

  for (uint j=0; j < curJ; j++) {

    double max_val = 0.0;
    uint arg_max = 100000;
    
    uint nNonIntegralVars = 0;

    for (uint i=0; i <= curI; i++) {

      double val = solution[j*(curI+1)+i];
      
      if (val > 0.01 && val < 0.99)
	nNonIntegralVars++;

      if (val > max_val) {

	max_val = val;
	arg_max = i;
      }
    }

    al.set(j+1,arg_max);
  }

  if (nNonIntegralVars > 0) {
    std::cerr << "ERROR: final solution is not integral. Exiting..." << std::endl;
    exit(1);
  }

  delete[] values;
  delete[] ibound2;
  delete[] wmax_fert;

  delete[] cost;
  delete[] var_lb;
  delete[] var_ub;
  delete[] rhs_lower;
  delete[] rhs_upper;

  delete[] row_idx;
  delete[] col_idx;
  delete[] matrix_entry;
  delete[] jcost_lower_bound;
  delete[] icost_lower_bound;
}

/************************************************************/
