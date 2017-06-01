#ifndef IC_MODIFICATION_HPP
#define IC_MODIFICATION_HPP

#include <src/tools/data_types/complex.hpp>
#include <memory>

#include <src/simulation/field/multilevelfield.hpp>
#include <src/cosmology/parameters.hpp>

namespace modifications{

	//! Abstract definition of a modification
	template<typename DataType, typename T=tools::datatypes::strip_complex <DataType>>
	class Modification : public std::enable_shared_from_this<Modification<DataType,T>>{
	private:
		T target;		/*!< Target to be achieved by the modification */

	protected:
		multilevelcontext::MultiLevelContextInformation<DataType> &underlying;
		const cosmology::CosmologicalParameters<T> &cosmology;
		std::vector<size_t> flaggedCells;		/*!< Region targeted by the modification */


	public:
		Modification(multilevelcontext::MultiLevelContextInformation<DataType> &underlying_,
								 const cosmology::CosmologicalParameters<T> &cosmology_): underlying(underlying_), cosmology(cosmology_){

			size_t finestlevel = this->underlying.getNumLevels() - 1;
			auto finestgrid = this->underlying.getGridForLevel(finestlevel);
			finestgrid.getFlaggedCells(flaggedCells);
		};

		//! Calculate modification value with a given field
		virtual T calculateCurrentValue(fields::MultiLevelField<DataType>* /* field */) = 0;

		T getTarget(){
			return target;
		}

		void setTarget(T target_){
			target = target_;
		}

	};

};




#endif
