import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .query import QuerySpec

#------------------------------------------------------------------------------
# Query is a abstract class that provides the interface to the Query API.  The
# implementation QueryImpl is in factbase.py but the interface is declared here
# so that we can deal with subqueries (imbedded within a membership clause (e.g,
# F.anum in fb.query(...)]).
# ------------------------------------------------------------------------------

class Query(abc.ABC):
    """An abstract class that defines the interface to the Clorm Query API v2.

    .. note::

       This new Query API replaces the old Select/Delete mechanism and offers
       many more features than the old API, especially allowing joins similar to
       an SQL join between tables.

       This interface is complete and unlikely to change - however it is being
       left open for the moment in case there is strong user feedback.

    ``Query`` objects cannot be constructed directly.

    Instead a ``Query`` object is returned by the :meth:`FactBase.query`
    function.  Queries can take a number of different forms but contain many of
    the components of a traditional SQL query. A predicate definition (as
    opposed to a predicate instance or fact) can be viewed as an SQL table and
    the parameters of a predicate can be viewed as the fields of the table.

    The simplest query must at least specify the predicate(s) to search
    for. This is specified as parameters to the :meth:`FactBase.query` function.
    Relating this to a traditional SQL query, the ``query`` clause can be viewed
    as an SQL ``FROM`` specification.

    The query is typicaly executed by iterating over the generator returned by
    the :meth:`Query.all` end-point.

    .. code-block:: python

       from clorm import FactBase, Predicate, IntegerField, StringField

       class Option(Predicate):
           oid = IntegerField
           name = StringField
           cost = IntegerField
           cat = StringField

       class Chosen(Predicate):
           oid = IntegerField

       fb=FactBase([Option(1,"Do A",200,"foo"),Option(2,"Do B",300,"bar"),
                    Option(3,"Do C",400,"foo"),Option(4,"Do D",300,"bar"),
                    Option(5,"Do E",200,"foo"),Option(6,"Do F",500,"bar"),
                    Chosen(1),Chosen(3),Chosen(4),Chosen(6)])

       q1 = fb.query(Chosen)     # Select all Chosen instances
       result = set(q1.all())
       assert result == set([Chosen(1),Chosen(3),Chosen(4),Chosen(6)])

    If there are multiple predicates involved in the search then the query must
    also contain a :meth:`Query.join` clause to specify the predicates
    parameters/fields to join.

    .. code-block:: python

       q2 = fb.query(Option,Chosen).join(Option.oid == Chosen.oid)

    .. note::

       As an aside, while a ``query`` clause typically consists of predicates,
       it can also contain predicate *aliases* created through the :func:`alias`
       function. This allows for queries with self joins to be specified.

    When a query contains multiple predicates the result will consist of tuples,
    where each tuple contains the facts matching the signature of predicates in
    the ``query`` clause. Mathematically the tuples are a subset of the
    cross-product over instances of the predicates; where the subset is
    determined by the ``join`` clause.

    .. code-block:: python

       result = set(q2.all())

       assert result == set([(Option(1,"Do A",200,"foo"),Chosen(1)),
                             (Option(3,"Do C",400,"foo"),Chosen(3)),
                             (Option(4,"Do D",300,"bar"),Chosen(4)),
                             (Option(6,"Do F",500,"bar"),Chosen(6))])

    A query can also contain a ``where`` clause as well as an ``order_by``
    clause. When the ``order_by`` clause contains a predicate path then by
    default it is ordered in ascending order. However, this can be changed to
    descending order with the :func:`desc` function modifier.

    .. code-block:: python

       from clorm import desc

       q3 = q2.where(Option.cost > 200).order_by(desc(Option.cost))

       result = list(q3.all())
       assert result == [(Option(6,"Do F",500,"bar"),Chosen(6)),
                         (Option(3,"Do C",400,"foo"),Chosen(3)),
                         (Option(4,"Do D",300,"bar"),Chosen(4))]

    The above code snippet highlights a feature of the query construction
    process. Namely, that these query construction functions can be chained and
    can also be used as the starting point for another query. Each construction
    function returns a modified copy of its parent. So in this example query
    ``q3`` is a modified version of query ``q2``.

    Returning tuples of facts is often not the most convenient output format and
    instead you may only be interested in specific predicates or parameters
    within each fact tuple. For example, in this running example it is
    unnecessary to return the ``Chosen`` facts. To provide the output in a more
    useful format a query can also contain a ``select`` clause that specifies
    the items to return. Essentially, this specifies a _projection_ over the
    elements of the result tuple.

    .. code-block:: python

       q4 = q3.select(Option)

       result = list(q4.all())
       assert result == [Option(6,"Do F",500,"bar"),
                         Option(3,"Do C",400,"foo"),
                         Option(4,"Do D",300,"bar")]

    A second mechanism for accessing the data in a more convenient format is to
    use a ``group_by`` clause. In this example, we may want to aggregate all the
    chosen options, for example to sum the costs, based on their membership of
    the ``"foo"`` and ``"bar"`` categories. The Clorm query API doesn't directly
    support aggregation functions, as you could do in SQL, so some additional
    Python code is required.

    .. code-block:: python

       q5 = q2.group_by(Option.cat).select(Option.cost)

       result = [(cat, sum(list(it))) for cat, it in q5.all()]
       assert result == [("bar",800), ("foo", 600)]

    The above are not the only options for a query. Some other query modifies
    include: :meth:`Query.distinct` to return distinct elements, and
    :meth:`Query.bind` to bind the value of any placeholders in the ``where``
    clause to specific values.

    A query is executed using a number of end-point functions. As already shown
    the main end-point is :meth:`Query.all` to return a generator for iterating
    over the results.

    Alternatively, if there is at least one element then to return the first
    result only (throwing an exception only if there are no elements) use the
    :meth:`Query.first` method.

    Or if there must be exactly one element (and to throw an exception
    otherwise) use :meth:`Query.singleton`.

    To count the elements of the result there is :meth:`Query.count`.

    Finally to delete all matching facts from the underlying FactBase use
    :meth:`Query.delete`.

    """

    #--------------------------------------------------------------------------
    # Add a join expression
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def join(self, *expressions):
        """Specifying how the predicate/tables in the query are to be joined.

        Joins are expressions that connect the predicates/tables of the
        query. They range from a pure SQL-like inner-join through to an
        unrestricted cross-product. The standard form is:

             ``<PredicatePath> <compop> <PredicatePath>``

        with the full cross-product expressed using a function:

             ``cross(<PredicatePath>,<PredicatePath>)``

        Every predicate/table in the query must be reachable to every other
        predicate/table through some form of join. For example, given predicate
        definitions ``F``, ``G``, ``H``, each with a field ``anum``:

              ``query = fb.query(F,G,H).join(F.anum == G.anum,cross(F,H))``

        generates an inner join between ``F`` and ``G``, but a full
        cross-product between ``F`` and ``H``.

        Finally, it is possible to perform self joins using the function
        ``alias`` that generates an alias for the predicate/table. For
        example:

              ``from clorm import alias``

              ``FA=alias(F)``
              ``query = fb.query(F,G,FA).join(F.anum == FA.anum,cross(F,G))``

        generates an inner join between ``F`` and itself, and a full
        cross-product between ``F`` and ``G``.

        Args:
          expressions: one or more join expressions.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Add an order_by expression
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def where(self, *expressions):
        """Sets a list of query conditions.

        The where clause consists of a single (or list) of simple/complex
        boolean and comparison expressions. This expression specifies a search
        criteria for matching facts within the corresponding ``FactBase``.

        Boolean expression are built from other boolean expression or a
        comparison expression. Comparison expressions are of the form:

               ``<PredicatePath> <compop>  <value>``

       where ``<compop>`` is a comparison operator such as ``==``, ``!=``, or
       ``<=`` and ``<value>`` is either a Python value or another predicate path
       object refering to a field of the same predicate or a placeholder.

        A placeholder is a special value that allows queries to be
        parameterised. A value can be bound to each placeholder. These
        placeholders are named ``ph1_``, ``ph2_``, ``ph3_``, and ``ph4_`` and
        correspond to the 1st to 4th arguments when the ``bind()`` function is
        called. Placeholders also allow for named arguments using the
        "ph_("<name>") function.

        Args:
          expressions: one or more comparison expressions.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Add an ordered expresison
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def ordered(self, *expressions):
        """Specify the natural/default ordering over results.

        Given a query over predicates 'P1...PN', the 'ordered()' flag provides a
        convenient way of specifying 'order_by(asc(P1),...,asc(PN))'.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Add an order_by expression
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def order_by(self, *expressions):
        """Specify an ordering over the results.

        Args:
          field_order: an ordering over fields

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Add a group_by expression
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def group_by(self, *expressions):
        """Specify a grouping over the results.

        The grouping specification is similar to an ordering specification but
        it modifies the behaviour of the query to return a pair of elements,
        where the first element of the pair is the group identifier (based on
        the specification) and the second element is an iterator over the
        matching elements.

        When both a ``group_by`` and ``order_by`` clause is provided the
        ``order_by`` clause is used the sort the elements within each matching
        group.

        Args:
          field_order: an ordering over fields to group by

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Specify a projection over the elements to output or delete
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def select(self,*outsig):
        """Provides a projection over the query result tuples.

        Mathematically the result tuples of a query are the cross-product over
        instances of the predicates. However, returning tuples of facts is often
        not the most convenient output format and instead you may only be
        interested in specific parameters/fields within each tuple of facts. The
        ``select`` clause specifies a _projection_ over each query result tuple.

        Each query result tuple is guaranteed to be distinct, since the query is
        a filtering over the cross-product of the predicate instances. However,
        the specification of a projection can result in information being
        discarded and can therefore cause the projected query results to no
        longer be distinct. To enforce uniqueness the :meth:`Query.distinct`
        flag can be specified. Essentially this is the same as an ``SQL SELECT
        DISTINCT ...`` statement.

        Note: when :meth:`Query.select` is used with the :meth:`Query.delete`
        end-point the `select` signature must specify predicates and not
        parameter/fields within the predicates.

        Finally, instead of a projection specification, a single Python
        `callable` object can be specified with input parameters matching the
        query signature. This callable object will then be called on each query
        tuple and the results will make up the modified query result. This
        provides the greatest flexibility in controlling the output of the
        query.

        Args:
           output_signature: the signature that defines the projection or a
                             callable object.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # The distinct flag
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def distinct(self):
        """Return only distinct elements in the query.

        This flag is only meaningful when combined with a :meth:`Query.select`
        clause that removes distinguishing elements from the tuples.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Ground - bind
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def bind(self,*args,**kwargs):
        """Bind placeholders to specific values.

        If the ``where`` clause has placeholders then these placeholders must be
        bound to actual values before the query can be executed.

        Args:
          *args: positional arguments corresponding to positional placeholders
          **kwargs: named arguments corresponding to named placeholders

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # The tuple flag
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def tuple(self):
        """Force returning a tuple even for singleton elements.

        In the general case the output signature of a query is a tuple;
        consisting either of a tuple of facts or a tuple of parameters/fields if
        a ``select`` projection has been specified.

        However, if the output signature is a singleton tuple then by default
        the API changes its behaviour and removes the tuple, returning only the
        element itself. This typically provides a much more useful and intutive
        interface. For example, if you want to perform a sum aggregation over
        the results of a query, aggregating over the value of a specific
        parameter/field, then specifying just that parameter in the ``select``
        clause allows you to simply pass the query generator to the standard
        Python ``sum()`` function without needing to perform a list
        comprehension to extract the value to be aggregated.

        If there is a case where this default behaviour is not wanted then
        specifying the ``tuple`` flag forces the query to always return a tuple
        of elements even if the output signature is a singleton tuple.

        Returns:
          Returns the modified copy of the query.

        """

    #--------------------------------------------------------------------------
    # Overide the default heuristic
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def heuristic(self, join_order):
        """Allows the query engine's query plan to be modified.

        This is an advanced option that can be used if the query is not
        performing as expected. For multi-predicate queries the order in which
        the joins are performed can affect performance. By default the Query API
        will try to optimise this join order based on the ``join`` expressions;
        with predicates with more restricted joins being higher up in the join
        order.

        This join order can be controlled explicitly by the ``fixed_join_order``
        heuristic function. Assuming predicate definitions ``F`` and ``G`` the
        query:

            ``from clorm import fixed_join_order``

            ``query=fb.query(F,G).heuristic(fixed_join_order(G,F)).join(...)``

        forces the join order to first be the ``G`` predicate followed by the
        ``F`` predicate.

        Args:
          join_order: the join order heuristic

        Returns:
          Returns the modified copy of the query.
        """

    #--------------------------------------------------------------------------
    # End points that do something useful
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Select to display all the output of the query
    # --------------------------------------------------------------------------
    @abc.abstractmethod
    def all(self):
        """Returns a generator that iteratively executes the query.

        Note. This call doesn't execute the query itself. The query is executed
        when iterating over the elements from the generator.

        Returns:
          Returns a generator that executes the query.

        """

    #--------------------------------------------------------------------------
    # Show the single element and throw an exception if there is more than one
    # --------------------------------------------------------------------------
    @abc.abstractmethod
    def singleton(self):
        """Return the single matching element.

           An exception is thrown if there is not exactly one matching element
           or a ``group_by()`` clause has been specified.

        Returns:
           Returns the single matching element (or throws an exception)

        """

    #--------------------------------------------------------------------------
    # Return the count of elements - Note: the behaviour of what is counted
    # changes if group_by() has been specified.
    # --------------------------------------------------------------------------
    @abc.abstractmethod
    def count(self):
        """Return the number of matching element.

           Typically the number of elements consist of the number of tuples
           produced by the cross-product of the predicates that match the
           criteria of the ``join()`` and ``where()`` clauses.

           However, if a ``select()`` projection and ``unique()`` flag is
           specified then the ``count()`` will reflect the modified the number
           of unique elements based on the projection of the query.

           Furthermore, if a ``group_by()`` clause is specified then ``count()``
           returns a generator that iterates over pairs where the first element
           of the pair is the group identifier and the second element is the
           number of matching elements within that group.

        Returns:
           Returns the number of matching elements

        """

    #--------------------------------------------------------------------------
    # Show the single element and throw an exception if there is more than one
    # --------------------------------------------------------------------------
    @abc.abstractmethod
    def first(self):
        """Return the first matching element.

           An exception is thrown if there are no one matching element or a
           ``group_by()`` clause has been specified.

        Returns:
           Returns the first matching element (or throws an exception)

        """

    #--------------------------------------------------------------------------
    # Delete a selection of fact
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def delete(self):
        """Delete matching facts from the ``FactBase()``.

        In the simple case of a query with no joins then ``delete()`` simply
        deletes the matching facts. If there is a join then the matches consist
        of tuples of facts. In this case ``delete()`` will remove all facts from
        the tuple. This behaviour can be modified by using a ``select()``
        projection clause that selects only specific predicates. Note: when
        combined with ``select()`` the output signature must specify predicates
        and not parameter/fields within predicates.

           An exception is thrown if a ``group_by()`` clause has been specified.

        Returns:
           Returns the number of facts deleted.

        """

    #--------------------------------------------------------------------------
    # For the user to see what the query plan looks like
    #--------------------------------------------------------------------------
    @abc.abstractmethod
    def query_plan(self,*args,**kwargs):
        """Return a query plan object outlining the query execution.

        A query plan outlines the query will be executed; the order of table
        joins, the searches based on indexing, and how the sorting is
        performed. This is useful for debugging if the query is not behaving as
        expected.

        Currently, there is no fixed specification for the query plan
        object. All the user can do is display it for reading and debugging
        purposes.

        Returns:
           Returns a query plan object that can be stringified.

        """

    #--------------------------------------------------------------------------
    # Internal API property
    #--------------------------------------------------------------------------
    @property
    def qspec(self) -> "QuerySpec":
        ...