#!/usr/bin/perl
package AI::ANN::Neuron;
BEGIN {
  $AI::ANN::Neuron::VERSION = '0.002';
}
# ABSTRACT: a neuron for an artificial neural network simulator

use strict;
use warnings;


sub new {
	my $proto = shift;
	my $class = ref($proto) || $proto;
	my $self = {};
	$self->{'id'} = shift;
	$self->{'inputs'} = shift;	#hashref
	$self->{'neurons'} = shift;	#hashref
	bless($self, $class);
	return $self;
}


sub ready {
	my $self = shift;
	my $inputs = shift;
	my $neurons = shift;

	foreach my $id (keys $self->{'inputs'}) { # Requires Perl 5.14 !!!
		unless ((not defined $self->{'inputs'}->{$id}) || 
				$self->{'inputs'}->{$id} == 0 || defined $inputs->[$id])
				{return 0}
		# This probably shouldn't ever happen, as it would be weird if our
		# inputs weren't available yet.
	}
	foreach my $id (keys $self->{'neurons'}) {
		unless ((not defined $self->{'neurons'}->{$id}) || 
				$self->{'neurons'}->{$id} == 0 || defined $neurons->{$id})
				{return 0}
		# First clause should be redundant, but you can never be too safe
	}
	return 1;
}


sub execute {
	my $self = shift;
	my $inputs = shift;
	my $neurons = shift;
	my $output = 0;
	foreach my $id (keys $self->{'inputs'}) {
		$output += ($self->{'inputs'}->{$id} || 0 ) * ($inputs->[$id] || 0);
	}
	foreach my $id (keys $self->{'neurons'}) {
		$output += ($self->{'neurons'}->{$id} || 0) * ($neurons->{$id} || 0);
	}
	return $output;
}


sub get_inputs {
	my $self = shift;
	return $self->{'inputs'};
}


sub get_neurons {
	my $self = shift;
	return $self->{'neurons'};
}

1;
		

__END__
=pod

=head1 NAME

AI::ANN::Neuron - a neuron for an artificial neural network simulator

=head1 VERSION

version 0.002

=head1 METHODS

=head2 new

AI::ANN::Neuron->new( $neuronid, {$inputid => $weight, ...}, {$neuronid => $weight} )

Weights may be whatever the user chooses. Note that packages that use this 
one may place their own restructions. Neurons and inputs are assumed to be 
zero-indexed.

=head2 ready

$neuron->ready( [$input0, $input1, ...], {$neuronid => $neuronvalue, ...} )

All inputs must be provided or you're insane.
Returns 1 if ready, 0 otherwise.

=head2 execute

$neuron->execute( [$input0, $input1, ...], {$neuronid => $neuronvalue, ...} )

All inputs must be provided or you're insane
Returns raw value (linear potential)

=head2 get_inputs

$neuron->get_inputs() 

Returns a hashref of the input values => weights

=head2 get_neurons

$neuron->get_neurons() 

Returns a hashref of the neuron values => weights

=head1 AUTHOR

Dan Collins <DCOLLINS@cpan.org>

=head1 COPYRIGHT AND LICENSE

This software is Copyright (c) 2011 by Dan Collins.

This is free software, licensed under:

  The GNU General Public License, Version 3, June 2007

=cut

